#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型推理 —— 特征 parquet/DataFrame → edge 列 → strict replay 输入。

用法:
  python qqq_btc/tools/run_inference.py \\
      --checkpoint checkpoints_qqq_net_edge_v2/best.pth \\
      --config qqq_btc/CONFIG/slow_feature_qqq_v2.json \\
      --input ~/train_data/quote_features_qqq_v2_norm/QQQ/.../1min/QQQ_2026-01.parquet \\
      --output /tmp/qqq_infer_2026-01.parquet \\
      --symbol-map qqq_btc/CONFIG/symbol_map.json \\
      --symbol QQQ
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qqq_btc.common.inference_tensors import SEQ_LEN, build_feature_maps, row_to_tensors
from qqq_btc.common.session_history import (
    DEFAULT_CARRYOVER_BARS,
    augment_with_session_carryover,
    prepend_carryover,
    real_bar_index,
    session_tail,
)
from qqq_btc.common.time_features import add_time_features, session_minute
from qqq_btc.common.regime_features import add_vix_regime_features
from qqq_btc.common.trend_features import add_trend_features, add_open30_features, add_spot_day_ret

logger = logging.getLogger("qqq_btc.inference")


def load_model(checkpoint: Path, config: dict, device):
    import torch

    from qqq_btc.model.backbone import DualStreamAlphaNet, resolve_embedding_caps
    caps = resolve_embedding_caps(config)
    model = DualStreamAlphaNet(config, caps)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in state:
                state = state[key]
                break
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def run_inference_df(
    df: pd.DataFrame,
    model,
    config: dict,
    stock_id: int,
    sector_id: int,
    device,
    batch_size: int = 256,
    use_carryover: bool = True,
    carryover_bars: int = DEFAULT_CARRYOVER_BARS,
) -> pd.DataFrame:
    import torch

    raw = df.copy().sort_values("timestamp").reset_index(drop=True)
    work = (
        augment_with_session_carryover(raw, carryover_bars=carryover_bars)
        if use_carryover
        else raw
    )
    work = work.copy().sort_values("timestamp").reset_index(drop=True)
    if "time_session_sin" not in work.columns:
        work = add_time_features(work)
    price_col = next((c for c in ("close", "price", "vwap") if c in work.columns), None)
    if price_col and (
        "trend_fit_ret_30m" not in work.columns or "spot_range_30m" not in work.columns
    ):
        work = add_trend_features(work, price_col=price_col)
    if price_col and "open30_ret" not in work.columns:
        work = add_open30_features(work, price_col=price_col)
    if "vix_proxy_close" in work.columns and "vix_reversal_count_30m" not in work.columns:
        work = add_vix_regime_features(work)
    if price_col and "spot_day_ret" not in work.columns:
        work = add_spot_day_ret(work, price_col=price_col)

    stock_map, option_map, n_stock, n_opt = build_feature_maps(config)
    if not work.empty:
        dow = pd.to_datetime(work["timestamp"].iloc[0]).dayofweek
    else:
        dow = 0

    n = len(work)
    preds = {
        "net_edge": np.zeros(n),
        "net_edge_q10": np.zeros(n),
        "net_edge_q50": np.zeros(n),
        "net_edge_q90": np.zeros(n),
        "call_net_edge": np.zeros(n),
        "put_net_edge": np.zeros(n),
        "straddle_net_edge": np.zeros(n),
        "best_side_put_prob": np.zeros(n),
        "best_side_none_prob": np.zeros(n),
        "best_side_call_prob": np.zeros(n),
        "best_bucket_id": np.full(n, -1),
        "best_bucket_conf": np.zeros(n),
        "spot_down_prob": np.zeros(n),
        "spot_flat_prob": np.zeros(n),
        "spot_up_prob": np.zeros(n),
        "spot_return_pred": np.zeros(n),
    }

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        xs, xo = [], []
        for i in range(start, end):
            a, b = row_to_tensors(work, i, stock_map, option_map, n_stock, n_opt)
            xs.append(a)
            xo.append(b)
        x_stk = torch.from_numpy(np.stack(xs)).to(device)
        x_opt = torch.from_numpy(np.stack(xo)).to(device)
        static = {
            "stock_id": torch.full((end - start,), stock_id, dtype=torch.long, device=device),
            "sector_id": torch.full((end - start,), sector_id, dtype=torch.long, device=device),
            "day_of_week": torch.full((end - start,), int(dow), dtype=torch.long, device=device),
        }
        with torch.no_grad():
            out = model(x_stk, x_opt, static)
        for k in (
            "net_edge",
            "net_edge_q10",
            "net_edge_q50",
            "net_edge_q90",
            "call_net_edge",
            "put_net_edge",
            "straddle_net_edge",
            "spot_return",
        ):
            if k in out:
                out_key = "spot_return_pred" if k == "spot_return" else k
                preds[out_key][start:end] = out[k].squeeze(-1).cpu().numpy()
        if "logits_best_side" in out:
            p_side = torch.softmax(out["logits_best_side"], dim=-1).cpu().numpy()
            preds["best_side_put_prob"][start:end] = p_side[:, 0]
            preds["best_side_none_prob"][start:end] = p_side[:, 1]
            preds["best_side_call_prob"][start:end] = p_side[:, 2]
        if "logits_best_bucket" in out:
            p_bucket = torch.softmax(out["logits_best_bucket"], dim=-1).cpu().numpy()
            bucket_id = np.argmax(p_bucket, axis=1)
            preds["best_bucket_id"][start:end] = np.where(bucket_id <= 7, bucket_id, -1)
            preds["best_bucket_conf"][start:end] = np.max(p_bucket, axis=1)
        if "logits_spot_dir" in out:
            p_spot = torch.softmax(out["logits_spot_dir"], dim=-1).cpu().numpy()
            preds["spot_down_prob"][start:end] = p_spot[:, 0]
            preds["spot_flat_prob"][start:end] = p_spot[:, 1]
            preds["spot_up_prob"][start:end] = p_spot[:, 2]

    for k, v in preds.items():
        work[k] = v
    if "session_bar" not in work.columns:
        work["session_bar"] = session_minute(work["timestamp"]).astype(int)

    if use_carryover and "_carryover" in work.columns:
        out = work.loc[~work["_carryover"].astype(bool)].copy()
        out = out.drop(columns=["_carryover"], errors="ignore")
    else:
        out = work
    return out.reset_index(drop=True)


def main() -> None:
    import torch

    from qqq_btc.common.seed_utils import resolve_seed, set_global_seed
    from qqq_btc.model.backbone import DualStreamAlphaNet, resolve_embedding_caps  # noqa: F401

    parser = argparse.ArgumentParser(description="qqq_btc 模型推理")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="qqq_btc/CONFIG/slow_feature_qqq_v2.json")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--symbol", default="QQQ")
    parser.add_argument("--symbol-map", default="qqq_btc/CONFIG/symbol_map.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=None, help="默认 42 或环境变量 QQQ_BTC_SEED")
    parser.add_argument(
        "--no-carryover",
        action="store_true",
        help="禁用前日 tail 拼入(默认开启,与 live carryover 一致)",
    )
    parser.add_argument("--carryover-bars", type=int, default=DEFAULT_CARRYOVER_BARS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    seed = set_global_seed(resolve_seed(args.seed), deterministic=True)
    logger.info("global seed=%s", seed)

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(args.symbol_map, "r", encoding="utf-8") as f:
        sym_map = json.load(f)[args.symbol]

    model = load_model(Path(args.checkpoint), config, device)
    df = pd.read_parquet(Path(args.input).expanduser())
    out = run_inference_df(
        df, model, config,
        stock_id=int(sym_map["stock_id"]),
        sector_id=int(sym_map["sector_id"]),
        device=device,
        batch_size=args.batch_size,
        use_carryover=not args.no_carryover,
        carryover_bars=args.carryover_bars,
    )
    Path(args.output).expanduser().parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    logger.info("written %s rows -> %s", len(out), args.output)


if __name__ == "__main__":
    main()
