#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parity 审计 —— 影子模式三张对账表。

1. 特征 parity: offline parquet vs live FCS 重算
2. 信号 parity: offline 推理 vs live 引擎同 bar 决策
3. Fill parity: 实盘 fill_spread_frac vs fill_model 假设

用法:
  python qqq_btc/tools/parity_audit.py feature \\
      --offline ~/train_data/.../1min/QQQ_2026-01.parquet \\
      --live ~/shadow/fcs_bars_2026-01.parquet \\
      --output /tmp/feature_parity.json

  python qqq_btc/tools/parity_audit.py fill \\
      --audit-log ~/shadow/fill_audit.csv \\
      --output /tmp/fill_parity.json

  python qqq_btc/tools/parity_audit.py exits \\
      --audit-log ~/shadow/fill_audit.csv \\
      --replay-trades /tmp/replay_trades.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qqq_btc.live.fcs_adapter import enrich_fcs_bars
from qqq_btc.qqq import config as qcfg

TIME_TREND = [
    "time_session_sin", "time_session_cos", "time_session_progress", "time_to_expiry_norm",
    "trend_fit_ret_30m", "trend_fit_r2_30m", "trend_fit_ret_120m", "trend_fit_r2_120m",
    "spot_range_30m", "trend_strength_30m",
    "day_range_pos", "drawdown_from_day_high", "drawup_from_day_low",
    "open30_ret", "open30_max_ret", "open30_peak_dd", "open30_reversal",
    "open30_range_pos", "bars_since_open30_high_norm",
]


def audit_features(offline: Path, live: Path, tol: float = 1e-4) -> dict:
    off = pd.read_parquet(offline)
    live_df = pd.read_parquet(live)
    live_df = enrich_fcs_bars(live_df)
    off = off.sort_values("timestamp").reset_index(drop=True)
    live_df = live_df.sort_values("timestamp").reset_index(drop=True)
    n = min(len(off), len(live_df))
    if n == 0:
        return {"error": "empty input"}

    cols = [c for c in TIME_TREND if c in off.columns and c in live_df.columns]
    if not cols:
        cols = [c for c in off.columns if c in live_df.columns and c.startswith(("options_", "close"))][:20]

    diffs = {}
    for c in cols:
        a = pd.to_numeric(off[c].iloc[:n], errors="coerce").fillna(0).values
        b = pd.to_numeric(live_df[c].iloc[:n], errors="coerce").fillna(0).values
        mad = float(np.mean(np.abs(a - b)))
        diffs[c] = {"mean_abs_diff": mad, "pass": mad <= tol}

    pass_rate = sum(1 for v in diffs.values() if v["pass"]) / max(1, len(diffs))
    return {"rows": n, "columns": diffs, "pass_rate": pass_rate, "tolerance": tol}


def audit_fill(audit_log: Path, target_frac: float = 0.775, tol: float = 0.05) -> dict:
    df = pd.read_csv(audit_log)
    col = "fill_spread_frac" if "fill_spread_frac" in df.columns else None
    if col is None:
        return {"error": "audit log 需含 fill_spread_frac 列"}
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return {"error": "no valid fill_spread_frac"}
    delta_col = "delta_frac" if "delta_frac" in df.columns else None
    out = {
        "n": int(len(s)),
        "median": float(s.median()),
        "p10": float(s.quantile(0.1)),
        "p90": float(s.quantile(0.9)),
        "target": target_frac,
        "median_within_tol": abs(float(s.median()) - target_frac) <= tol,
        "pass": abs(float(s.median()) - target_frac) <= tol,
    }
    if delta_col:
        d = pd.to_numeric(df[delta_col], errors="coerce").dropna()
        if len(d):
            out["delta_frac_median"] = float(d.median())
    return out


def audit_exit_reasons(audit_log: Path, replay_reasons: Path | None = None) -> dict:
    """
    从 fill_audit CLOSE 行的 exit_reason 统计分布;
    可选与 strict replay 导出的 exit_reason 计数对拍。
    """
    df = pd.read_csv(audit_log)
    if "action" in df.columns:
        closes = df[df["action"].astype(str).str.upper() == "CLOSE"]
    else:
        closes = df
    col = "exit_reason" if "exit_reason" in closes.columns else None
    if col is None:
        return {"error": "audit log 需含 exit_reason 列(CLOSE 成交)"}
    reasons = closes[col].fillna("").astype(str)
    reasons = reasons[reasons.str.len() > 0]
    if reasons.empty:
        return {"error": "no CLOSE exit_reason rows", "n_close": int(len(closes))}
    live_counts = reasons.value_counts(normalize=True).to_dict()
    out = {"n_close": int(len(reasons)), "live_distribution": {k: float(v) for k, v in live_counts.items()}}

    if replay_reasons is not None and replay_reasons.exists():
        rep = pd.read_csv(replay_reasons)
        rc = rep["exit_reason"] if "exit_reason" in rep.columns else rep.iloc[:, 0]
        rep_counts = rc.fillna("").astype(str)
        rep_counts = rep_counts[rep_counts.str.len() > 0].value_counts(normalize=True).to_dict()
        out["replay_distribution"] = {k: float(v) for k, v in rep_counts.items()}
        keys = set(live_counts) | set(rep_counts)
        l1 = sum(abs(live_counts.get(k, 0) - rep_counts.get(k, 0)) for k in keys)
        out["distribution_l1"] = float(l1)
        out["pass"] = l1 <= 0.35
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="qqq_btc parity 审计")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_feat = sub.add_parser("feature")
    p_feat.add_argument("--offline", required=True)
    p_feat.add_argument("--live", required=True)
    p_feat.add_argument("--output", default=None)
    p_feat.add_argument("--tol", type=float, default=1e-4)

    p_fill = sub.add_parser("fill")
    p_fill.add_argument("--audit-log", required=True)
    p_fill.add_argument("--output", default=None)
    p_fill.add_argument("--target-frac", type=float, default=qcfg.FILL_MODEL.entry_frac)

    p_exit = sub.add_parser("exits", help="CLOSE exit_reason 分布 vs replay")
    p_exit.add_argument("--audit-log", required=True, help="fill_audit.csv")
    p_exit.add_argument("--replay-trades", default=None, help="replay trades CSV 含 exit_reason")
    p_exit.add_argument("--output", default=None)

    args = parser.parse_args()
    if args.cmd == "feature":
        report = audit_features(Path(args.offline), Path(args.live), tol=args.tol)
    elif args.cmd == "fill":
        report = audit_fill(Path(args.audit_log), target_frac=args.target_frac)
    else:
        report = audit_exit_reasons(
            Path(args.audit_log),
            Path(args.replay_trades) if args.replay_trades else None,
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
