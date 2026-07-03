#!/usr/bin/env python3
"""
Phase 1 bidirectional audit: normal vs oracle vs put_only side attribution.

Joins alpha_logs to locked ATM option quotes (when history DB exists) and reports:
  - missed put/call edge by day_type
  - gate-style attribution (direction from alpha sign vs oracle)
  - tradable minute ratio

Example:
    cd New_Pro/baseline_qqq
    python tools/bidirectional_phase1_audit.py --dates 20260302 --mode synthetic
    python tools/bidirectional_phase1_audit.py --dates 20260302,20260303 --hist-dir ~/quant_project/data/history_sqlite_1s
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bidirectional_regime import DayType, oracle_side_from_returns, resolve_day_type, resolve_micro_regime
from utils.audit_alpha_executable_edge import (
    AuditConfig,
    _date_to_db,
    _parse_locked_atm_quote,
    _safe_float,
)

REPORT_DIR = ROOT / "reports" / "bidirectional_phase1"


@dataclass
class Phase1Summary:
    dates: List[str] = field(default_factory=list)
    mode: str = "normal"
    n_minutes: int = 0
    tradable_ratio: float = 0.0
    oracle_call_share: float = 0.0
    oracle_put_share: float = 0.0
    model_call_share: float = 0.0
    model_put_share: float = 0.0
    side_agreement: float = 0.0
    missed_put_edge_mean: float = 0.0
    missed_call_edge_mean: float = 0.0
    by_day_type: Dict[str, dict] = field(default_factory=dict)


def _synthetic_minutes(n: int = 390) -> pd.DataFrame:
    """Deterministic synthetic path for CI / dry-run without history DB."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        day_roc = np.sin(i / 60) * 0.006
        roc_5m = day_roc * 0.8 + rng.normal(0, 0.0003)
        snap = rng.normal(0, 0.001)
        call_ret = max(0, day_roc * 8 + snap * 3 + rng.normal(0, 0.002))
        put_ret = max(0, -day_roc * 8 - snap * 3 + rng.normal(0, 0.002))
        alpha = (call_ret - put_ret) * 0.5
        spread = float(rng.uniform(0.03, 0.10))
        ctx = {
            "qqq_day_roc": day_roc,
            "stock_roc": roc_5m,
            "snap_roc": snap,
            "alpha": alpha,
            "options_vw_spread": spread,
            "options_iv_momentum": 0.05,
        }
        day_type = resolve_day_type(ctx).value
        micro = resolve_micro_regime(ctx)
        oracle = oracle_side_from_returns(call_ret, put_ret)
        model = 1 if alpha > 0.015 else (-1 if alpha < -0.015 else 0)
        rows.append({
            "minute": i,
            "day_type": day_type,
            "micro_regime": micro,
            "call_ret": call_ret,
            "put_ret": put_ret,
            "alpha": alpha,
            "oracle_side": oracle,
            "model_side": model,
            "spread": spread,
        })
    return pd.DataFrame(rows)


def _load_alpha_minutes(hist_dir: Path, date: str, cfg: AuditConfig) -> pd.DataFrame:
    db = _date_to_db(hist_dir, date)
    if not db.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(db))
    try:
        df = pd.read_sql_query(
            """
            SELECT ts, symbol, alpha, vol_z, options_vw_spread, options_iv_momentum,
                   locked_buckets_json, stock_price, roc_5m, snap_roc
            FROM alpha_logs
            WHERE symbol = 'QQQ'
            ORDER BY ts
            """,
            conn,
        )
    finally:
        conn.close()
    if df.empty:
        return df

    delay = int(cfg.delay_seconds)
    hold = int(cfg.holding_seconds)
    out_rows = []
    for _, row in df.iterrows():
        put_bid, put_ask, _, call_bid, call_ask, _ = _parse_locked_atm_quote(
            str(row.get("locked_buckets_json", "") or "")
        )
        if not np.isfinite(put_ask) or not np.isfinite(call_ask):
            continue
        call_ret = (call_bid - call_ask) / call_ask if call_ask > 0 else np.nan
        put_ret = (put_bid - put_ask) / put_ask if put_ask > 0 else np.nan
        alpha = _safe_float(row.get("alpha"))
        ctx = {
            "qqq_day_roc": _safe_float(row.get("roc_5m")),
            "stock_roc": _safe_float(row.get("roc_5m")),
            "snap_roc": _safe_float(row.get("snap_roc")),
            "alpha": alpha,
            "options_vw_spread": _safe_float(row.get("options_vw_spread")),
            "options_iv_momentum": _safe_float(row.get("options_iv_momentum")),
        }
        out_rows.append({
            "date": date,
            "ts": row["ts"],
            "day_type": resolve_day_type(ctx).value,
            "micro_regime": resolve_micro_regime(ctx),
            "call_ret": call_ret,
            "put_ret": put_ret,
            "alpha": alpha,
            "oracle_side": oracle_side_from_returns(call_ret, put_ret),
            "model_side": 1 if alpha > cfg.min_edge else (-1 if alpha < -cfg.min_edge else 0),
            "spread": _safe_float(row.get("options_vw_spread")),
        })
    return pd.DataFrame(out_rows)


def summarize(df: pd.DataFrame, *, mode: str, dates: List[str]) -> Phase1Summary:
    s = Phase1Summary(dates=dates, mode=mode, n_minutes=len(df))
    if df.empty:
        return s
    s.tradable_ratio = float((df["micro_regime"] == "tradable").mean())
    oracle = df["oracle_side"]
    model = df["model_side"]
    s.oracle_call_share = float((oracle == 1).mean())
    s.oracle_put_share = float((oracle == -1).mean())
    s.model_call_share = float((model == 1).mean())
    s.model_put_share = float((model == -1).mean())
    agree = (oracle != 0) & (model == oracle)
    s.side_agreement = float(agree.mean())
    missed_put = df[(oracle == -1) & (model != -1)]
    missed_call = df[(oracle == 1) & (model != 1)]
    s.missed_put_edge_mean = float(missed_put["put_ret"].mean()) if len(missed_put) else 0.0
    s.missed_call_edge_mean = float(missed_call["call_ret"].mean()) if len(missed_call) else 0.0
    for dt, g in df.groupby("day_type"):
        s.by_day_type[str(dt)] = {
            "n": int(len(g)),
            "oracle_put_share": float((g["oracle_side"] == -1).mean()),
            "model_put_share": float((g["model_side"] == -1).mean()),
            "side_agreement": float(((g["oracle_side"] != 0) & (g["model_side"] == g["oracle_side"])).mean()),
        }
    return s


def write_report(summary: Phase1Summary, df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "PHASE1_SUMMARY.md"
    lines = [
        "# Phase 1 Bidirectional Audit Summary",
        "",
        f"- dates: `{', '.join(summary.dates)}`",
        f"- mode: `{summary.mode}`",
        f"- minutes: **{summary.n_minutes}**",
        f"- tradable_ratio: **{summary.tradable_ratio:.1%}**",
        f"- oracle call/put share: **{summary.oracle_call_share:.1%}** / **{summary.oracle_put_share:.1%}**",
        f"- model call/put share: **{summary.model_call_share:.1%}** / **{summary.model_put_share:.1%}**",
        f"- side agreement (oracle vs model): **{summary.side_agreement:.1%}**",
        f"- missed put edge (mean ret when oracle=put, model≠put): **{summary.missed_put_edge_mean:.4f}**",
        f"- missed call edge: **{summary.missed_call_edge_mean:.4f}**",
        "",
        "## By day_type",
        "",
    ]
    for dt, stats in summary.by_day_type.items():
        lines.append(
            f"- `{dt}`: n={stats['n']} | oracle_put={stats['oracle_put_share']:.1%} "
            f"| model_put={stats['model_put_share']:.1%} | agree={stats['side_agreement']:.1%}"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "phase1_summary.json").write_text(
        json.dumps(summary.__dict__, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not df.empty:
        df.to_csv(out_dir / "phase1_minutes.csv", index=False)
    return md


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Phase 1 bidirectional audit")
    p.add_argument("--dates", default="", help="Comma-separated YYYYMMDD (optional for synthetic)")
    p.add_argument("--hist-dir", default=str(Path.home() / "quant_project/data/history_sqlite_1s"))
    p.add_argument("--mode", choices=("normal", "oracle", "put_only", "synthetic"), default="normal")
    p.add_argument("--out-dir", default=str(REPORT_DIR))
    args = p.parse_args(argv)

    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    frames = []
    if args.mode == "synthetic" or not dates:
        frames.append(_synthetic_minutes())
        dates = dates or ["synthetic"]
    else:
        cfg = AuditConfig()
        hist = Path(args.hist_dir)
        for d in dates:
            part = _load_alpha_minutes(hist, d, cfg)
            if not part.empty:
                frames.append(part)
        if not frames:
            print("No history data; falling back to synthetic minutes.")
            frames.append(_synthetic_minutes())
            dates = ["synthetic"]

    df = pd.concat(frames, ignore_index=True)
    if args.mode == "put_only":
        df["model_side"] = np.where(df["alpha"] < -0.015, -1, 0)
    elif args.mode == "oracle":
        df["model_side"] = df["oracle_side"]

    summary = summarize(df, mode=args.mode, dates=dates)
    path = write_report(summary, df, Path(args.out_dir))
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
