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

from qqq_btc.tools.run_inference import load_model, run_inference_df
from qqq_btc.common.replay_harness import run_strict_replay
from qqq_btc.qqq import config as qcfg

logger = logging.getLogger("qqq_btc.eval_test")


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


def merge_1m_5m(path_1m: Path, path_5m: Path, feats_5m: list[str]) -> pd.DataFrame:
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import torch

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(args.symbol_map, "r", encoding="utf-8") as f:
        sym = json.load(f)[args.symbol]

    root = Path(args.feature_root).expanduser() / args.symbol / "regular" / "09:30-16:00"
    files_1m = sorted((root / "1min").glob("*.parquet"))
    if not files_1m:
        raise SystemExit(f"no 1min files under {root / '1min'}")

    _, feats_5m = _feat_names_by_res(config)
    model = load_model(Path(args.checkpoint), config, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_parts = []
    for f1 in files_1m:
        f5 = root / "5min" / f1.name
        logger.info("infer %s", f1.name)
        df = merge_1m_5m(f1, f5, feats_5m)
        pred = run_inference_df(
            df,
            model,
            config,
            stock_id=int(sym["stock_id"]),
            sector_id=int(sym["sector_id"]),
            device=device,
            use_carryover=True,
        )
        # label_pipeline 已写入 exec_*;仅缺列时从 1m 期权回补
        need_exec = not {"exec_call_bid", "exec_call_ask"}.issubset(pred.columns)
        if need_exec:
            pred = attach_exec_quotes(
                pred,
                Path(args.option_1m_root),
                args.symbol,
                call_bucket=qcfg.TRADE_BUCKET_ID,
                put_bucket=0,
            )
        else:
            # 清理历史错误后缀列
            for leg in ("call", "put"):
                for side in ("bid", "ask"):
                    base = f"exec_{leg}_{side}"
                    for suf in ("_x", "_y"):
                        alt = base + suf
                        if alt in pred.columns and base not in pred.columns:
                            pred[base] = pred[alt]
                    if base in pred.columns and f"exec_{leg}_mid" not in pred.columns:
                        pass
            if "exec_call_mid" not in pred.columns and "exec_call_bid" in pred.columns:
                pred["exec_call_mid"] = (
                    pd.to_numeric(pred["exec_call_bid"], errors="coerce")
                    + pd.to_numeric(pred["exec_call_ask"], errors="coerce")
                ) / 2.0
        all_parts.append(pred)

    full = pd.concat(all_parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    infer_path = out_dir / "test_infer.parquet"
    full.to_parquet(infer_path, index=False)
    logger.info("written %s rows -> %s", len(full), infer_path)

    metrics = label_metrics(full)
    logger.info("test label metrics: %s", metrics)

    result = run_strict_replay(
        full,
        qcfg.FILL_MODEL,
        qcfg.REPLAY,
        qcfg.EXIT_RAILS,
        edge_col="net_edge",
        edge_q10_col=qcfg.EDGE_Q10_COL,
        call_edge_col=qcfg.CALL_EDGE_COL,
        put_edge_col=qcfg.PUT_EDGE_COL,
        put_gate_col=qcfg.PUT_GATE_COL,
    )
    summary = result.summary(position_frac=qcfg.REPLAY.position_frac)
    summary["label_metrics"] = metrics
    summary["n_rows"] = int(len(full))
    summary["checkpoint"] = str(args.checkpoint)
    summary["session_entry_start_bar"] = qcfg.REPLAY.session_entry_start_bar
    summary["session_entry_end_bar"] = qcfg.REPLAY.session_entry_end_bar

    summary_path = out_dir / "replay_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    trades_path = out_dir / "replay_trades.parquet"
    result.trades_frame().to_parquet(trades_path, index=False)
    logger.info("replay summary: %s", summary)
    logger.info("trades -> %s", trades_path)


if __name__ == "__main__":
    main()
