#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exit rails 重标定工具 —— 用数据反推止损/档位,替代沿用 9DTE 的手拍数值。

方法:MAE/MFE(最大不利/有利偏移)分析。
对【会入场的 bar】按统一 FillModel 成交,向前滚动 max_hold_bars,
记录 mid MTM 的 ROI 路径,统计:

  - 赢单 MAE q05/q01 → soft/hard 止损
  - 赢单 MFE 分位 → ladder ratchet 档位
  - 标签 horizon(默认 5 bar)处 ROI → early_stop
  - 15 bar 未走强 → time_stop_min_roi
  - 大 MFE → trailing / flash / disaster

按时段分桶(0DTE 上午/午间/下午),输出每桶建议 + 保守合并的 ExitRailsConfig 片段。

用法(真实数据):
    python qqq_btc/tools/calibrate_rails.py \\
      --parquet <含 exec_call_* + net_edge 的特征文件> \\
      --entry-threshold 0.015 \\
      --out rails_suggestion.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qqq_btc.common.fill_model import OptionSpreadFillModel
from qqq_btc.common.time_features import session_minute

DEFAULT_TOD_BUCKETS = {
    "morning": (0, 120),      # 09:30-11:30
    "midday": (120, 270),     # 11:30-14:00
    "afternoon": (270, 390),  # 14:00-16:00
}

DEFAULT_HORIZON_BARS = (5, 15)


def spread_pct(bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    mid = (bid + ask) / 2.0
    return np.where(mid > 0, (ask - bid) / mid, np.inf)


def entry_bar_mask(
    df: pd.DataFrame,
    *,
    edge_col: str = "net_edge",
    entry_threshold: float = 0.015,
    max_spread_pct: float = 0.06,
    edge_q10_col: Optional[str] = "net_edge_q10",
    require_q10_positive: bool = False,
    edge_q10_floor: Optional[float] = -0.20,
    session_entry_end_bar: Optional[int] = 330,
    bid_col: str = "exec_call_bid",
    ask_col: str = "exec_call_ask",
) -> np.ndarray:
    """与 replay 入场门控对齐的布尔掩码(每个 bar 是否「会尝试入场」)。"""
    n = len(df)
    mask = np.ones(n, dtype=bool)

    if edge_col in df.columns:
        edge = pd.to_numeric(df[edge_col], errors="coerce").to_numpy(dtype=np.float64)
        mask &= np.isfinite(edge) & (edge >= entry_threshold)
    else:
        mask &= False

    if edge_q10_col and edge_q10_col in df.columns:
        q10 = pd.to_numeric(df[edge_q10_col], errors="coerce").to_numpy(dtype=np.float64)
        if require_q10_positive:
            mask &= np.isfinite(q10) & (q10 > 0)
        elif edge_q10_floor is not None:
            mask &= np.isfinite(q10) & (q10 > float(edge_q10_floor))

    if "timestamp" in df.columns and session_entry_end_bar is not None:
        sbar = session_minute(df["timestamp"]).to_numpy()
        mask &= sbar <= int(session_entry_end_bar)

    if bid_col in df.columns and ask_col in df.columns:
        bid = pd.to_numeric(df[bid_col], errors="coerce").to_numpy(dtype=np.float64)
        ask = pd.to_numeric(df[ask_col], errors="coerce").to_numpy(dtype=np.float64)
        sp = spread_pct(bid, ask)
        mask &= np.isfinite(sp) & (sp <= max_spread_pct)

    return mask


def compute_roi_paths(
    df: pd.DataFrame,
    fill_model: OptionSpreadFillModel,
    max_hold_bars: int = 30,
    bid_col: str = "exec_call_bid",
    ask_col: str = "exec_call_ask",
    mid_col: str = "exec_call_mid",
    horizon_bars: Sequence[int] = DEFAULT_HORIZON_BARS,
    entry_mask: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    每个潜在入场 bar 的持有期统计:final_roi / mae / mfe / session_min / roi_h*。
    ROI 相对 entry fill 价(0.775 点差位),MTM 用 mid。
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n = len(df)
    if n <= max_hold_bars:
        return pd.DataFrame()

    bid = pd.to_numeric(df[bid_col], errors="coerce").to_numpy(dtype=np.float64)
    ask = pd.to_numeric(df[ask_col], errors="coerce").to_numpy(dtype=np.float64)
    if mid_col in df.columns:
        mid = pd.to_numeric(df[mid_col], errors="coerce").to_numpy(dtype=np.float64)
    else:
        mid = (bid + ask) / 2.0

    entry_fill = fill_model.entry_fill(bid, ask)
    valid_len = n - max_hold_bars

    fwd = sliding_window_view(mid[1:], window_shape=max_hold_bars)[:valid_len]
    entry = entry_fill[:valid_len]

    with np.errstate(invalid="ignore", divide="ignore"):
        roi_path = fwd / entry[:, None] - 1.0

    ok = (
        np.isfinite(entry) & (entry > 0)
        & np.all(np.isfinite(fwd) & (fwd > 0), axis=1)
    )

    out: Dict[str, np.ndarray] = {
        "final_roi": roi_path[:, -1],
        "mae": np.nanmin(roi_path, axis=1),
        "mfe": np.nanmax(roi_path, axis=1),
        "valid": ok,
    }
    for h in horizon_bars:
        if 1 <= h <= max_hold_bars:
            out[f"roi_h{h}"] = roi_path[:, h - 1]
            out[f"mae_h{h}"] = np.nanmin(roi_path[:, :h], axis=1)
            out[f"mfe_h{h}"] = np.nanmax(roi_path[:, :h], axis=1)

    if "timestamp" in df.columns:
        out["session_min"] = session_minute(df["timestamp"]).to_numpy()[:valid_len]
    else:
        out["session_min"] = np.zeros(valid_len, dtype=np.int64)

    if entry_mask is not None:
        out["is_entry_bar"] = entry_mask[:valid_len]
    else:
        out["is_entry_bar"] = np.ones(valid_len, dtype=bool)

    frame = pd.DataFrame(out)
    frame = frame[frame["valid"] & frame["is_entry_bar"]].drop(
        columns=["valid", "is_entry_bar"]
    ).reset_index(drop=True)
    return frame


def _round4(x: float) -> float:
    return round(float(x), 4)


def _suggest_early_stop(
    seg: pd.DataFrame,
    horizon: int = 5,
) -> Optional[dict]:
    col = f"roi_h{horizon}"
    if col not in seg.columns:
        return None
    winners = seg[seg["final_roi"] > 0]
    if len(winners) < 20:
        return None
    # 只有 ~5% 赢单在 horizon 处 ROI 低于此阈值 → 误杀率可控
    roi = float(winners[col].quantile(0.05))
    roi = min(roi, -0.02)  # 至少 -2%,避免过紧
    roi = max(roi, -0.15)
    losers = seg[seg["final_roi"] <= 0]
    loser_cut = (
        float((losers[col] <= roi).mean())
        if len(losers) >= 10
        else None
    )
    return {
        "early_stop_bars": horizon,
        "early_stop_roi": _round4(roi),
        "winner_roi_h_q05": _round4(float(winners[col].quantile(0.05))),
        "loser_cut_rate_at_horizon": loser_cut,
    }


def _suggest_time_stop(seg: pd.DataFrame, horizon: int = 15) -> dict:
    col = f"roi_h{horizon}"
    winners = seg[seg["final_roi"] > 0]
    if len(winners) >= 20 and col in winners.columns:
        # 赢单在 horizon 处 ROI 的 q25 * 0.8,夹在 3%-8%
        raw = float(winners[col].quantile(0.25)) * 0.8
        min_roi = _round4(max(0.03, min(0.08, raw)))
    else:
        min_roi = 0.05
    return {
        "time_stop_bars": horizon,
        "time_stop_min_roi": min_roi,
    }


def suggest_rails(
    paths: pd.DataFrame,
    tod_buckets: dict = None,
    *,
    early_stop_horizon: int = 5,
    time_stop_horizon: int = 15,
    ladder_keep: Tuple[Tuple[float, float], ...] = (
        (0.5, 0.60), (0.75, 0.65), (0.9, 0.70),
    ),
) -> dict:
    """从 ROI 路径统计生成分时段 rails 建议。"""
    tod_buckets = tod_buckets or DEFAULT_TOD_BUCKETS
    report: dict = {}
    for name, (lo, hi) in tod_buckets.items():
        seg = paths[(paths["session_min"] >= lo) & (paths["session_min"] < hi)]
        if len(seg) < 50:
            report[name] = {"samples": int(len(seg)), "note": "样本不足,不给建议"}
            continue
        winners = seg[seg["final_roi"] > 0]
        if len(winners) < 30:
            report[name] = {"samples": int(len(seg)), "note": "赢单不足,不给建议"}
            continue

        soft = float(winners["mae"].quantile(0.05))
        hard = float(winners["mae"].quantile(0.01))
        mfe_q = winners["mfe"].quantile([0.5, 0.75, 0.9])

        tiers = []
        for q, keep in ladder_keep:
            trig = _round4(float(mfe_q.loc[q]))
            tiers.append([trig, _round4(trig * keep)])

        early = _suggest_early_stop(seg, horizon=early_stop_horizon)
        time_stop = _suggest_time_stop(seg, horizon=time_stop_horizon)

        mfe90 = float(mfe_q.loc[0.9])
        first_trig = tiers[0][0] if tiers else 0.08

        bucket = {
            "samples": int(len(seg)),
            "entry_samples": int(len(seg)),
            "win_rate": float((seg["final_roi"] > 0).mean()),
            "suggested_soft_stop_roi": _round4(soft),
            "suggested_hard_stop_roi": _round4(hard),
            "suggested_ladder_ratchet": tiers,
            "suggested_trailing_trigger_roi": _round4(max(mfe90, 0.25)),
            "suggested_trailing_keep_ratio": 0.65,
            "suggested_flash_trigger_roi": _round4(max(first_trig, 0.06)),
            "suggested_flash_exit_roi": _round4(max(first_trig * 0.4, 0.02)),
            "suggested_disaster_stop_roi": _round4(min(hard * 2.0, -0.20)),
            **time_stop,
        }
        if early:
            bucket.update({
                "suggested_early_stop_bars": early["early_stop_bars"],
                "suggested_early_stop_roi": early["early_stop_roi"],
                "early_stop_winner_roi_h_q05": early["winner_roi_h_q05"],
                "early_stop_loser_cut_rate": early["loser_cut_rate_at_horizon"],
            })
        report[name] = bucket

    merged = merge_bucket_suggestions(report)
    if merged:
        report["_merged_conservative"] = merged
    return report


def merge_bucket_suggestions(report: dict) -> dict:
    """
    跨时段保守合并:止损取最紧(least negative),ladder 取 trigger 最低档,
    供单一 ExitRailsConfig 占位(实盘可按 session 切换 schedule)。
    """
    buckets = [
        v for k, v in report.items()
        if not k.startswith("_") and "suggested_soft_stop_roi" in v
    ]
    if not buckets:
        return {}

    # 30min 持有:取各时段中位宽度(过紧会把噪声当止损;过宽用 tick disaster 兜底)
    soft = float(np.median([v["suggested_soft_stop_roi"] for v in buckets]))
    hard = float(np.median([v["suggested_hard_stop_roi"] for v in buckets]))
    soft = max(min(soft, -0.10), -0.35)
    hard = max(min(hard, soft - 0.03), -0.45)

    ladders = [v["suggested_ladder_ratchet"] for v in buckets if v.get("suggested_ladder_ratchet")]
    merged_ladder: list = []
    if ladders:
        n_tiers = min(len(x) for x in ladders)
        for i in range(n_tiers):
            trig = min(x[i][0] for x in ladders)
            floor = min(x[i][1] for x in ladders)
            merged_ladder.append([_round4(trig), _round4(floor)])

    early_bars = [
        v["suggested_early_stop_bars"]
        for v in buckets if "suggested_early_stop_bars" in v
    ]
    early_roi = [
        v["suggested_early_stop_roi"]
        for v in buckets if "suggested_early_stop_roi" in v
    ]

    time_bars = int(np.median([v["time_stop_bars"] for v in buckets]))
    time_min = float(np.median([v["time_stop_min_roi"] for v in buckets]))

    trailing_trig = min(v["suggested_trailing_trigger_roi"] for v in buckets)
    disaster = min(v["suggested_disaster_stop_roi"] for v in buckets)

    flash_trig = min(v["suggested_flash_trigger_roi"] for v in buckets)
    flash_exit = min(v["suggested_flash_exit_roi"] for v in buckets)

    cfg = {
        "soft_stop_roi": _round4(soft),
        "hard_stop_roi": _round4(hard),
        "early_stop_bars": int(np.median(early_bars)) if early_bars else None,
        "early_stop_roi": _round4(max(early_roi)) if early_roi else None,
        "time_stop_bars": time_bars,
        "time_stop_min_roi": _round4(time_min),
        "trailing_trigger_roi": _round4(trailing_trig),
        "trailing_keep_ratio": 0.65,
        "ladder": merged_ladder,
        "flash_trigger_roi": _round4(flash_trig),
        "flash_exit_roi": _round4(flash_exit),
        "disaster_stop_roi": _round4(disaster),
        "note": "保守合并:止损取各桶最紧;填回 qqq_btc/qqq/config.py EXIT_RAILS 前请 strict replay 验证",
    }
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="从报价+edge 数据反推 exit rails 参数")
    parser.add_argument("--parquet", required=True, help="含 exec_call_bid/ask/mid + net_edge 的特征 parquet")
    parser.add_argument("--max-hold-bars", type=int, default=30)
    parser.add_argument("--entry-frac", type=float, default=0.775)
    parser.add_argument("--entry-threshold", type=float, default=0.015)
    parser.add_argument("--edge-col", default="net_edge")
    parser.add_argument("--max-spread-pct", type=float, default=0.06)
    parser.add_argument("--require-q10", action="store_true", help="要求 net_edge_q10 > 0")
    parser.add_argument("--edge-q10-col", default="net_edge_q10")
    parser.add_argument("--edge-q10-floor", type=float, default=-0.20,
                        help="q10 下限门控(与 ReplayConfig.edge_q10_floor 一致)")
    parser.add_argument("--session-entry-end-bar", type=int, default=330)
    parser.add_argument("--no-entry-filter", action="store_true", help="不过滤,用全 bar(不推荐)")
    parser.add_argument("--early-stop-horizon", type=int, default=15, help="孵化期/early_stop bar")
    parser.add_argument("--time-stop-horizon", type=int, default=30)
    parser.add_argument("--out", default=None, help="JSON 输出路径(默认 stdout)")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    fm = OptionSpreadFillModel(entry_frac=args.entry_frac, exit_frac=args.entry_frac)

    entry_mask = None
    if not args.no_entry_filter:
        entry_mask = entry_bar_mask(
            df,
            edge_col=args.edge_col,
            entry_threshold=args.entry_threshold,
            max_spread_pct=args.max_spread_pct,
            edge_q10_col=args.edge_q10_col,
            require_q10_positive=args.require_q10,
            edge_q10_floor=None if args.require_q10 else args.edge_q10_floor,
            session_entry_end_bar=args.session_entry_end_bar,
        )
        n_entry = int(entry_mask.sum())
        if n_entry == 0:
            print(
                f"无入场 bar(edge>={args.entry_threshold}, col={args.edge_col})。"
                "若仅有 label 无预测,可 --no-entry-filter 或换 --edge-col label_return_fwd_net",
                file=sys.stderr,
            )
            return

    paths = compute_roi_paths(
        df,
        fm,
        max_hold_bars=args.max_hold_bars,
        horizon_bars=(args.early_stop_horizon, args.time_stop_horizon),
        entry_mask=entry_mask,
    )
    if paths.empty:
        print("无有效 ROI 路径,请检查 exec_call_* 列与 entry 过滤条件。")
        return

    report = suggest_rails(
        paths,
        early_stop_horizon=args.early_stop_horizon,
        time_stop_horizon=args.time_stop_horizon,
    )
    meta = {
        "parquet": str(args.parquet),
        "entry_filter": not args.no_entry_filter,
        "entry_threshold": args.entry_threshold,
        "edge_col": args.edge_col,
        "path_samples": int(len(paths)),
    }
    report["_meta"] = meta

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"已写入 {args.out} ({len(paths)} 条入场路径)")
    else:
        print(text)


if __name__ == "__main__":
    main()
