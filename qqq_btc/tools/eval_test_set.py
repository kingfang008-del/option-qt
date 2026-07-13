#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test 集推理 + L1 strict replay。

1min 特征与 5min regime 列对齐后推理;从 databento 1m 期权回补 CALL/PUT ATM 盘口做 fill 回放。

用法:
  python qqq_btc/tools/eval_test_set.py \\
      --checkpoint checkpoints_qqq_v2/best.pth \\
      --feature-root ~/train_data/quote_features_test \\
      --option-1m-root /mnt/s990/data/raw_1m/options_databento \\
      --output-dir /tmp/qqq_btc_test_eval
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

import importlib

from qqq_btc.tools.run_inference import load_model, run_inference_df
from qqq_btc.common.replay_harness import run_strict_replay
from qqq_btc.qqq import config as qcfg

logger = logging.getLogger("qqq_btc.eval_test")


def _load_strategy_config(module_path: str | None):
    """加载策略规则模块;默认 1DTE 族 ``qqq_btc.qqq.config``。"""
    if not module_path:
        return qcfg
    mod = importlib.import_module(module_path)
    logger.info("strategy-config=%s profile=%s", module_path, getattr(mod, "PROFILE", "?"))
    return mod


def _feat_names_by_res(config: dict) -> tuple[list[str], list[str]]:
    f1, f5 = [], []
    for f in config.get("features", []):
        name = f["name"]
        res = f.get("resolution", "1min")
        if res == "5min":
            f5.append(name)
        else:
            f1.append(name)
    return f1, f5


def merge_1m_5m(
    path_1m: Path,
    path_5m: Path,
    feats_5m: list[str],
    *,
    causal_5m: bool = False,
    bar_minutes: int = 5,
) -> pd.DataFrame:
    """合并 1min 主表与 5min 特征。

    causal_5m=False(历史默认):merge_asof(backward) 直接用 5min 左标签时间戳。
      若 5min close 含桶末价格,则在桶内 1–4 分钟会读到未收盘才确定的值(前视)。
    causal_5m=True:把 5min 可用时刻平移到桶结束(timestamp + bar_minutes),
      只有桶走完后才 asof 得到该 bar —— 与实盘 buffer/等收盘一致。
    """
    df1 = pd.read_parquet(path_1m)
    df1["timestamp"] = pd.to_datetime(df1["timestamp"])
    if not path_5m.exists():
        logger.warning("no 5min file: %s", path_5m)
        return df1.sort_values("timestamp").reset_index(drop=True)

    df5 = pd.read_parquet(path_5m)
    df5["timestamp"] = pd.to_datetime(df5["timestamp"])
    cols = [c for c in feats_5m if c in df5.columns]
    if not cols:
        return df1.sort_values("timestamp").reset_index(drop=True)

    # 去掉 1min 上错误尺度的 5min 列，用真正 5min 文件 asof 回填
    keep = [c for c in df1.columns if c not in cols]
    base = df1[keep].sort_values("timestamp")
    right = df5[["timestamp"] + cols].sort_values("timestamp").drop_duplicates(
        subset=["timestamp"], keep="last"
    )
    # tz align
    if base["timestamp"].dt.tz is None and right["timestamp"].dt.tz is not None:
        base["timestamp"] = base["timestamp"].dt.tz_localize(right["timestamp"].dt.tz)
    elif base["timestamp"].dt.tz is not None and right["timestamp"].dt.tz is None:
        right["timestamp"] = right["timestamp"].dt.tz_localize(base["timestamp"].dt.tz)
    elif (
        base["timestamp"].dt.tz is not None
        and right["timestamp"].dt.tz is not None
        and str(base["timestamp"].dt.tz) != str(right["timestamp"].dt.tz)
    ):
        right["timestamp"] = right["timestamp"].dt.tz_convert(base["timestamp"].dt.tz)

    if causal_5m:
        right = right.copy()
        right["timestamp"] = right["timestamp"] + pd.Timedelta(minutes=int(bar_minutes))
        logger.info(
            "causal_5m: delay 5min features by %dmin before asof (%s)",
            int(bar_minutes),
            path_5m.name,
        )

    out = pd.merge_asof(base, right, on="timestamp", direction="backward")
    return out.sort_values("timestamp").reset_index(drop=True)


def _align_ts(left: pd.Series, right: pd.Series) -> tuple[pd.Series, pd.Series]:
    l, r = left.copy(), right.copy()
    if l.dt.tz is None and r.dt.tz is not None:
        l = l.dt.tz_localize(r.dt.tz)
    elif l.dt.tz is not None and r.dt.tz is None:
        r = r.dt.tz_localize(l.dt.tz)
    elif l.dt.tz is not None and r.dt.tz is not None and str(l.dt.tz) != str(r.dt.tz):
        r = r.dt.tz_convert(l.dt.tz)
    return l, r


def drop_embedded_exec_columns(df: pd.DataFrame) -> pd.DataFrame:
    """去掉特征/label_pipeline 内嵌的 exec_* 列,避免回放用陈旧盘口。"""
    drop = [c for c in df.columns if c.startswith("exec_")]
    return df.drop(columns=drop) if drop else df


def attach_exec_quotes(
    df: pd.DataFrame,
    option_root: Path,
    symbol: str,
    call_bucket: int = 2,
    put_bucket: int = 0,
    tolerance: str | None = "5min",
) -> pd.DataFrame:
    """
    从日频 1m 期权文件按 bucket 回补 exec_* 盘口。

    tolerance:merge_asof 回看容差。报价断档超过该时长的分钟置 NaN,
    replay 对 NaN 盘口不开仓 —— 数据缺洞时段自动"当天不交易"
    (2026 年 bucket0 覆盖劣化,陈旧价成交会伪造 PUT 腿收益)。None=旧行为。
    """
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    ts_ny = out["timestamp"]
    if ts_ny.dt.tz is None:
        ts_ny = ts_ny.dt.tz_localize("America/New_York")
    else:
        ts_ny = ts_ny.dt.tz_convert("America/New_York")
    dates = ts_ny.dt.date.unique()

    quote_parts = []
    for d in dates:
        day = pd.Timestamp(d).strftime("%Y-%m-%d")
        fp = option_root / symbol / f"{symbol}_{day}.parquet"
        if not fp.exists():
            continue
        opt = pd.read_parquet(fp)
        if "bucket_id" not in opt.columns:
            continue
        opt["timestamp"] = pd.to_datetime(opt["timestamp"])
        for bucket, prefix in ((call_bucket, "exec_call"), (put_bucket, "exec_put")):
            sub = opt[opt["bucket_id"] == bucket][["timestamp", "bid", "ask"]].copy()
            if sub.empty and bucket == call_bucket:
                sub = opt[opt["bucket_id"] == 3][["timestamp", "bid", "ask"]].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
            sub = sub.rename(columns={"bid": f"{prefix}_bid", "ask": f"{prefix}_ask"})
            quote_parts.append((prefix, sub))

    if not quote_parts:
        logger.warning("未找到任何期权盘口，replay 将无法成交")
        return out

    tol = pd.Timedelta(tolerance) if tolerance is not None else None
    out = out.sort_values("timestamp")
    # 各腿独立 merge_asof:断档超容差 → NaN → replay 该腿不开仓
    for prefix in ("exec_call", "exec_put"):
        parts = [s for p, s in quote_parts if p == prefix]
        if not parts:
            continue
        quotes = pd.concat(parts, ignore_index=True).sort_values("timestamp")
        quotes = quotes.drop_duplicates("timestamp", keep="last")
        out["timestamp"], quotes["timestamp"] = _align_ts(out["timestamp"], quotes["timestamp"])
        out = pd.merge_asof(
            out.sort_values("timestamp"),
            quotes,
            on="timestamp",
            direction="backward",
            tolerance=tol,
        )
    for leg in ("call", "put"):
        b, a = f"exec_{leg}_bid", f"exec_{leg}_ask"
        if b in out.columns and a in out.columns:
            mid = (out[b] + out[a]) / 2.0
            out[f"exec_{leg}_spread_pct"] = np.where(
                (out[b] > 0) & (out[a] > out[b]),
                (out[a] - out[b]) / mid.replace(0, np.nan),
                np.nan,
            )
    return out


def label_metrics(df: pd.DataFrame) -> dict:
    if "label_return_fwd_net" not in df.columns or "net_edge" not in df.columns:
        return {}
    r = pd.to_numeric(df["label_return_fwd_net"], errors="coerce")
    p = pd.to_numeric(df["net_edge"], errors="coerce")
    m = r.notna() & p.notna()
    r, p = r[m], p[m]
    if len(r) < 50 or r.std() < 1e-12 or p.std() < 1e-12:
        return {"ic": 0.0, "n": int(len(r))}
    ic = float(p.corr(r, method="spearman"))
    n_top = max(10, int(len(p) * 0.05))
    top = pd.DataFrame({"p": p.values, "r": r.values}).nlargest(n_top, "p")
    out = {
        "ic": ic,
        "n": int(len(r)),
        "top5_mean_net": float(top["r"].mean()),
        "top5_hit": float((top["r"] > 0).mean()),
        "pred_std": float(p.std()),
        "label_std": float(r.std()),
    }
    if "net_edge_q10" in df.columns:
        q10 = pd.to_numeric(df["net_edge_q10"], errors="coerce")[m]
        out["q10_coverage"] = float((r < q10).mean())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="qqq_btc test eval")
    parser.add_argument("--checkpoint", default="checkpoints_qqq_v2/best.pth")
    parser.add_argument("--config", default="qqq_btc/CONFIG/slow_feature_qqq_v2.json")
    parser.add_argument("--feature-root", default="~/train_data/quote_features_test")
    parser.add_argument("--option-1m-root", default="/mnt/s990/data/raw_1m/options_databento")
    parser.add_argument("--symbol", default="QQQ")
    parser.add_argument("--symbol-map", default="qqq_btc/CONFIG/symbol_map.json")
    parser.add_argument("--output-dir", default="/tmp/qqq_btc_test_eval")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None, help="默认 42 或环境变量 QQQ_BTC_SEED")
    parser.add_argument("--call-bucket", type=int, default=None)
    parser.add_argument("--put-bucket", type=int, default=0)
    parser.add_argument(
        "--strategy-config",
        default=None,
        help="策略规则模块,如 qqq_btc.qqq.config_true_0dte;默认 qqq_btc.qqq.config(1DTE族)",
    )
    parser.add_argument(
        "--frozen-norm",
        default=None,
        help="日冻结 normalizer .npz;feature-root 须为 quote_features_raw,与 FCS 同文件",
    )
    parser.add_argument(
        "--infer-parquet",
        default=None,
        help="若已有 test_infer.parquet,跳过推理直接 replay(省时间)",
    )
    parser.add_argument(
        "--causal-5m",
        action="store_true",
        help="5min 特征仅在桶结束后可用(去掉桶内前视);默认关闭以复现旧口径",
    )
    parser.add_argument(
        "--live-replay",
        action="store_true",
        help="用 LIVE_REPLAY(entry_delay=0) 而非 REPLAY",
    )
    parser.add_argument(
        "--put-gate-raw5",
        default=None,
        help="用 raw 5min vix_level asof/因果覆盖 put_gate(阈值在 raw 标尺);"
        "目录或文件。与 --causal-5m 同时时按因果可用时刻 asof",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import torch

    from qqq_btc.common.seed_utils import resolve_seed, set_global_seed

    scfg = _load_strategy_config(args.strategy_config)
    if args.call_bucket is None:
        args.call_bucket = int(scfg.TRADE_BUCKET_ID)

    seed = set_global_seed(resolve_seed(args.seed), deterministic=True)
    logger.info("global seed=%s", seed)

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(args.symbol_map, "r", encoding="utf-8") as f:
        sym = json.load(f)[args.symbol]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.infer_parquet:
        full = pd.read_parquet(Path(args.infer_parquet).expanduser())
        logger.info("reuse infer parquet %s rows=%d", args.infer_parquet, len(full))
    else:
        root = Path(args.feature_root).expanduser() / args.symbol / "regular" / "09:30-16:00"
        files_1m = sorted((root / "1min").glob("*.parquet"))
        if not files_1m:
            raise SystemExit(f"no 1min files under {root / '1min'}")

        _, feats_5m = _feat_names_by_res(config)
        model = load_model(Path(args.checkpoint), config, device)

        all_parts = []
        for f1 in files_1m:
            f5 = root / "5min" / f1.name
            logger.info("infer %s causal_5m=%s", f1.name, args.causal_5m)
            df = merge_1m_5m(f1, f5, feats_5m, causal_5m=bool(args.causal_5m))
            pred = run_inference_df(
                df,
                model,
                config,
                stock_id=int(sym["stock_id"]),
                sector_id=int(sym["sector_id"]),
                device=device,
                use_carryover=True,
                frozen_norm=Path(args.frozen_norm).expanduser() if args.frozen_norm else None,
            )
            # 回放成交一律从 databento 1m 重新 attach(5min 容差),不用特征内嵌 exec_*
            pred = drop_embedded_exec_columns(pred)
            pred = attach_exec_quotes(
                pred,
                Path(args.option_1m_root),
                args.symbol,
                call_bucket=args.call_bucket,
                put_bucket=args.put_bucket,
            )
            all_parts.append(pred)

        full = pd.concat(all_parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        infer_path = out_dir / "test_infer.parquet"
        full.to_parquet(infer_path, index=False)
        logger.info("written %s rows -> %s", len(full), infer_path)

    # 可选:put_gate 用 raw 5min(阈值标定在 raw 标尺;frozen vix 会压死 early_vix)
    put_gate_col = scfg.PUT_GATE_COL
    if args.put_gate_raw5:
        raw_path = Path(args.put_gate_raw5).expanduser()
        if raw_path.is_dir():
            fps = sorted(raw_path.glob("*.parquet"))
        else:
            fps = [raw_path]
        raw_frames = []
        for fp in fps:
            d = pd.read_parquet(fp, columns=["timestamp", "vix_level"])
            d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
            raw_frames.append(d)
        raw5 = pd.concat(raw_frames, ignore_index=True).sort_values("timestamp")
        raw5 = raw5.drop_duplicates("timestamp", keep="last")
        if args.causal_5m:
            raw5 = raw5.copy()
            raw5["timestamp"] = raw5["timestamp"] + pd.Timedelta(minutes=5)
        full = full.copy()
        full["timestamp"] = pd.to_datetime(full["timestamp"], utc=True)
        m = pd.merge_asof(
            full[["timestamp"]].reset_index(drop=True),
            raw5.rename(columns={"vix_level": "_put_gate_raw5"}),
            on="timestamp",
            direction="backward",
        )
        full["put_gate_raw5"] = m["_put_gate_raw5"].to_numpy()
        put_gate_col = "put_gate_raw5"
        logger.info(
            "put_gate overridden with raw5 (%s) causal=%s ge0.6=%.3f",
            raw_path,
            args.causal_5m,
            float((full["put_gate_raw5"] >= 0.6).mean()),
        )

    metrics = label_metrics(full)
    logger.info("test label metrics: %s", metrics)

    replay_cfg = scfg.LIVE_REPLAY if args.live_replay else scfg.REPLAY
    result = run_strict_replay(
        full,
        scfg.FILL_MODEL,
        replay_cfg,
        scfg.EXIT_RAILS,
        edge_col="net_edge",
        edge_q10_col=scfg.EDGE_Q10_COL,
        call_edge_col=scfg.CALL_EDGE_COL,
        put_edge_col=scfg.PUT_EDGE_COL,
        put_gate_col=put_gate_col,
    )
    summary = result.summary(position_frac=replay_cfg.position_frac)
    summary["label_metrics"] = metrics
    summary["n_rows"] = int(len(full))
    summary["checkpoint"] = str(args.checkpoint)
    summary["seed"] = int(seed)
    summary["causal_5m"] = bool(args.causal_5m)
    summary["live_replay"] = bool(args.live_replay)
    summary["put_gate_col"] = put_gate_col
    if args.put_gate_raw5:
        summary["put_gate_raw5"] = str(Path(args.put_gate_raw5).expanduser())
    summary["strategy_config"] = args.strategy_config or "qqq_btc.qqq.config"
    summary["profile"] = getattr(scfg, "PROFILE", "1dte_family")
    summary["session_entry_start_bar"] = scfg.REPLAY.session_entry_start_bar
    summary["session_entry_end_bar"] = scfg.REPLAY.session_entry_end_bar
    summary["call_bucket"] = int(args.call_bucket)
    summary["put_bucket"] = int(args.put_bucket)
    summary["exit_rails"] = {
        "hard_stop_roi": scfg.EXIT_RAILS.hard_stop_roi,
        "max_hold_bars": scfg.EXIT_RAILS.max_hold_bars,
        "vol_scale_ref": scfg.EXIT_RAILS.vol_scale_ref,
        "time_stop_bars": scfg.EXIT_RAILS.time_stop_bars,
    }
    if args.frozen_norm:
        summary["frozen_norm"] = str(Path(args.frozen_norm).expanduser())

    summary_path = out_dir / "replay_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    trades_path = out_dir / "replay_trades.parquet"
    result.trades_frame().to_parquet(trades_path, index=False)
    logger.info("replay summary: %s", summary)
    logger.info("trades -> %s", trades_path)


if __name__ == "__main__":
    main()
