#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
同日信号 diff —— strict replay(SIGNAL) vs live 路径(ENTER/immediate) vs dry-run CSV。

用法:
  python qqq_btc/tools/signal_diff_day.py \\
    --parquet /tmp/qqq_btc_test_eval_v4/test_infer.parquet \\
    --date 2026-06-02 \\
    --output /tmp/signal_diff_20260602.json

  # 若有 dry-run 导出的信号 CSV:
  python qqq_btc/tools/signal_diff_day.py \\
    --parquet /tmp/qqq_btc_test_eval_v4/test_infer.parquet \\
    --date 2026-06-02 \\
    --dry-run-signals ~/quant_project/shadow/signals_20260602.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qqq_btc.common.signal_collect import (
    collect_decision_signals,
    collect_live_sim_signals,
    collect_replay_signals,
    diff_signal_frames,
    load_dry_run_signals,
)
from qqq_btc.qqq import config as qcfg


def _default_parquet() -> Path | None:
    for p in (
        Path("/tmp/qqq_btc_test_eval_v4/test_infer.parquet"),
        _REPO / "data" / "test_infer.parquet",
    ):
        if p.exists():
            return p
    return None


def run_day_diff(
    *,
    parquet: Path,
    date: str,
    dry_run_signals: Path | None = None,
    output: Path | None = None,
    tolerance_bars: int = 1,
) -> dict:
    df = pd.read_parquet(parquet)
    replay_sig = collect_replay_signals(
        df,
        replay_cfg=qcfg.REPLAY,
        warmup_through_day=date,
        target_day=date,
        signal_kinds=("SIGNAL",),
        source="strict_replay",
        signal_only=False,
    )
    live_sig = collect_live_sim_signals(
        df,
        warmup_through_day=date,
        target_day=date,
    )
    decision_replay = collect_decision_signals(df, warmup_through_day=date, target_day=date, replay_cfg=qcfg.REPLAY)
    decision_live = collect_decision_signals(df, warmup_through_day=date, target_day=date, replay_cfg=qcfg.LIVE_REPLAY)

    diff_sim = diff_signal_frames(replay_sig, live_sig, time_tolerance_bars=tolerance_bars)
    diff_decision = diff_signal_frames(decision_replay, decision_live, time_tolerance_bars=0)

    report: dict = {
        "date": date,
        "parquet": str(parquet),
        "replay_signals": replay_sig.to_dict("records"),
        "live_sim_signals": live_sig.to_dict("records"),
        "decision_replay": decision_replay.to_dict("records"),
        "decision_live": decision_live.to_dict("records"),
        "replay_vs_live_sim": diff_sim,
        "decision_replay_vs_live": diff_decision,
    }

    if dry_run_signals is not None and dry_run_signals.exists():
        dry_sig = load_dry_run_signals(dry_run_signals)
        dry_day = dry_sig[dry_sig["date"] == str(pd.Timestamp(date).date())].copy()
        dry_day["ts"] = dry_day["ts"].astype(str)
        report["dry_run_signals"] = dry_day.to_dict("records")
        report["replay_vs_dry_run"] = diff_signal_frames(
            replay_sig, dry_day, time_tolerance_bars=tolerance_bars
        )
        report["live_sim_vs_dry_run"] = diff_signal_frames(
            live_sig, dry_day, time_tolerance_bars=0
        )

    s = diff_sim["summary"]
    ds = diff_decision["summary"]
    lines = [
        f"=== Signal diff {date} ===",
        "",
        "[Decision layer — signal_only, no position]",
        f"  replay cfg: {ds.get('n_replay', 0)}  live cfg: {ds.get('n_live', 0)}  matched: {ds.get('n_matched', 0)}",
        f"  match rate: {ds.get('match_rate_replay', 0):.1%}",
        "",
        "[Operational — full replay strict SIGNAL vs live ENTER]",
        f"  strict replay SIGNAL: {s.get('n_replay', 0)}",
        f"  live sim ENTER:       {s.get('n_live', 0)}",
        f"  matched (±{tolerance_bars} bar): {s.get('n_matched', 0)}",
        f"  match rate replay: {s.get('match_rate_replay', 0):.1%}",
        f"  match rate live:   {s.get('match_rate_live', 0):.1%}",
    ]
    if diff_sim["replay_only"]:
        lines.append(f"\nreplay-only ({len(diff_sim['replay_only'])}):")
        for row in diff_sim["replay_only"][:10]:
            lines.append(f"  sb={row.get('session_bar')} {row.get('leg')} edge={row.get('edge'):.4f}")
    if diff_sim["live_only"]:
        lines.append(f"\nlive-only ({len(diff_sim['live_only'])}):")
        for row in diff_sim["live_only"][:10]:
            lines.append(f"  sb={row.get('session_bar')} {row.get('leg')} edge={row.get('edge'):.4f}")

    text = "\n".join(lines)
    print(text)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        csv_base = output.with_suffix("")
        replay_sig.to_csv(f"{csv_base}_replay.csv", index=False)
        live_sig.to_csv(f"{csv_base}_live_sim.csv", index=False)
        print(f"\nWrote {output}")
        print(f"Wrote {csv_base}_replay.csv / {csv_base}_live_sim.csv")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="strict replay vs live 同日入场信号 diff")
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--date", required=True, help="YYYY-MM-DD (America/New_York 交易日)")
    parser.add_argument("--dry-run-signals", default=None, help="dry-run 信号 CSV(可选)")
    parser.add_argument("--output", default=None, help="JSON 报告路径")
    parser.add_argument("--tolerance-bars", type=int, default=1)
    args = parser.parse_args()

    pq = Path(args.parquet).expanduser() if args.parquet else _default_parquet()
    if pq is None or not pq.exists():
        print("ERROR: --parquet 未指定或文件不存在", file=sys.stderr)
        sys.exit(1)

    dry = Path(args.dry_run_signals).expanduser() if args.dry_run_signals else None
    out = Path(args.output).expanduser() if args.output else None
    report = run_day_diff(
        parquet=pq,
        date=args.date,
        dry_run_signals=dry,
        output=out,
        tolerance_bars=int(args.tolerance_bars),
    )
    ds = report["decision_replay_vs_live"]["summary"]
    decision_ok = ds.get("n_replay", 0) == ds.get("n_live", 0) == ds.get("n_matched", 0)
    sys.exit(0 if decision_ok else 2)


if __name__ == "__main__":
    main()
