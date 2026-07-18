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
from dataclasses import replace
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
    first_entry_diff,
    load_dry_run_signals,
)
from qqq_btc.common.exit_lifecycle import (
    audit_fill_model_declared,
    collect_replay_exits,
    date_scoped_fill_audit_path,
    diff_exit_lifecycle,
    first_exit_diff,
    load_fill_audit_exits,
)
from qqq_btc.qqq import config as qcfg


def _parity_replay_cfg():
    """历史 honest KPI / 流式统一使用 immediate LIVE 与 q10=-0.2。"""
    return replace(qcfg.LIVE_REPLAY, edge_q10_floor=-0.2)


def _limit_session_bar(df: pd.DataFrame, max_session_bar: int) -> pd.DataFrame:
    if df.empty or "session_bar" not in df.columns:
        return df.copy()
    return df[df["session_bar"].astype(int) <= int(max_session_bar)].copy()


def _default_parquet() -> Path | None:
    for p in (
        Path("/tmp/qqq_btc_test_eval_v4/test_infer.parquet"),
        _REPO / "data" / "test_infer.parquet",
    ):
        if p.exists():
            return p
    return None


def _default_dry_run_signals(date: str) -> Path | None:
    p = Path.home() / "quant_project" / "shadow" / f"signals_{date}.csv"
    return p if p.exists() else None


def _default_se_alpha_signals(date: str) -> Path | None:
    p = Path.home() / "quant_project" / "shadow" / f"se_alpha_{date}.csv"
    return p if p.exists() else None


def load_se_alpha_decisions(
    path: str | pd.PathLike,
    *,
    date: str,
    max_session_bar: int | None = None,
) -> pd.DataFrame:
    raw = pd.read_csv(path)
    empty_cols = ["ts", "leg", "edge", "session_bar", "threshold", "source", "kind", "date"]
    if raw.empty:
        return pd.DataFrame(columns=empty_cols)
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(raw.get("timestamp", raw.get("ts")), utc=True, errors="coerce")
    out["session_bar"] = pd.to_numeric(raw.get("session_bar"), errors="coerce")
    out["net_edge"] = pd.to_numeric(raw.get("net_edge_raw"), errors="coerce")
    out[qcfg.CALL_EDGE_COL] = pd.to_numeric(raw.get("call_edge"), errors="coerce")
    out[qcfg.PUT_EDGE_COL] = pd.to_numeric(raw.get("put_edge"), errors="coerce")
    out[qcfg.EDGE_Q10_COL] = pd.to_numeric(raw.get("net_edge_q10"), errors="coerce")
    out[qcfg.PUT_GATE_COL] = pd.to_numeric(raw.get("vix_level"), errors="coerce")
    for col in (
        "spot_day_ret",
        "spot_ret_5bar",
        "trend_fit_ret_30m",
        "trend_fit_r2_30m",
        "spot_range_30m",
        "open30_max_ret",
        "open30_peak_dd",
        "vix_reversal_count_30m",
    ):
        if col in raw.columns:
            out[col] = pd.to_numeric(raw[col], errors="coerce")
    out = out.dropna(subset=["timestamp", "session_bar", "net_edge"]).sort_values("timestamp")
    target_day = str(pd.Timestamp(date).date())
    out = out[out["timestamp"].dt.tz_convert("America/New_York").dt.date.astype(str) == target_day].copy()
    if max_session_bar is not None:
        out = _limit_session_bar(out, int(max_session_bar))
    if out.empty:
        return pd.DataFrame(columns=empty_cols)
    return collect_decision_signals(
        out,
        warmup_from_day=target_day,
        warmup_through_day=target_day,
        target_day=target_day,
        replay_cfg=_parity_replay_cfg(),
    )


def run_day_diff(
    *,
    parquet: Path,
    date: str,
    dry_run_signals: Path | None = None,
    se_alpha_signals: Path | None = None,
    output: Path | None = None,
    tolerance_bars: int = 1,
    warmup_from_day: str | None = None,
    warmup_through_day: str | None = None,
    max_session_bar: int | None = None,
    fill_audit_path: Path | None = None,
) -> dict:
    df = pd.read_parquet(parquet)
    through = warmup_through_day or date
    parity_cfg = _parity_replay_cfg()
    replay_sig = collect_replay_signals(
        df,
        replay_cfg=parity_cfg,
        warmup_from_day=warmup_from_day,
        warmup_through_day=through,
        target_day=date,
        signal_kinds=("SIGNAL",),
        source="strict_replay",
        signal_only=False,
    )
    live_sig = collect_live_sim_signals(
        df,
        replay_cfg=parity_cfg,
        warmup_from_day=warmup_from_day,
        warmup_through_day=through,
        target_day=date,
    )
    decision_replay = collect_decision_signals(
        df,
        warmup_from_day=warmup_from_day,
        warmup_through_day=through,
        target_day=date,
        replay_cfg=parity_cfg,
    )
    decision_live = collect_decision_signals(
        df,
        warmup_from_day=warmup_from_day,
        warmup_through_day=through,
        target_day=date,
        replay_cfg=parity_cfg,
    )

    if max_session_bar is not None:
        max_sb = int(max_session_bar)
        replay_sig = _limit_session_bar(replay_sig, max_sb)
        live_sig = _limit_session_bar(live_sig, max_sb)
        decision_replay = _limit_session_bar(decision_replay, max_sb)
        decision_live = _limit_session_bar(decision_live, max_sb)

    diff_sim = diff_signal_frames(replay_sig, live_sig, time_tolerance_bars=tolerance_bars)
    diff_decision = diff_signal_frames(decision_replay, decision_live, time_tolerance_bars=0)

    report: dict = {
        "date": date,
        "warmup_from_day": warmup_from_day,
        "warmup_through_day": through,
        "parquet": str(parquet),
        "max_session_bar": int(max_session_bar) if max_session_bar is not None else None,
        "replay_signals": replay_sig.to_dict("records"),
        "live_sim_signals": live_sig.to_dict("records"),
        "decision_replay": decision_replay.to_dict("records"),
        "decision_live": decision_live.to_dict("records"),
        "replay_vs_live_sim": diff_sim,
        "decision_replay_vs_live": diff_decision,
    }

    if dry_run_signals is None:
        dry_run_signals = _default_dry_run_signals(date)
    if se_alpha_signals is None:
        se_alpha_signals = _default_se_alpha_signals(date)

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

    if dry_run_signals is not None and dry_run_signals.exists():
        dry_sig = load_dry_run_signals(dry_run_signals)
        dry_day = dry_sig[dry_sig["date"] == str(pd.Timestamp(date).date())].copy()
        if max_session_bar is not None and "session_bar" in dry_day.columns:
            dry_day = dry_day[dry_day["session_bar"].astype(int) <= int(max_session_bar)].copy()
        dry_day["ts"] = dry_day["ts"].astype(str)
        report["dry_run_signals"] = dry_day.to_dict("records")
        report["dry_run_path"] = str(dry_run_signals)
        # 诊断:无仓 decision SIGNAL vs OMS PASS(条数天然不等,不作主门禁)
        report["replay_vs_dry_run"] = diff_signal_frames(
            decision_replay, dry_day, time_tolerance_bars=0
        )
        # 主门禁:占仓感知 live_sim 全量 ENTER vs dry PASS
        report["live_sim_vs_dry_run"] = diff_signal_frames(
            live_sig, dry_day, time_tolerance_bars=int(tolerance_bars)
        )
        # 首笔对拍(OMS 层验收口径)
        report["first_entry_vs_dry_run"] = first_entry_diff(
            live_sig, dry_day, time_tolerance_bars=int(tolerance_bars)
        )
        fe = report["first_entry_vs_dry_run"]["summary"]
        ls = report["live_sim_vs_dry_run"]["summary"]
        lines.extend(
            [
                "",
                f"[OMS first-entry — live_sim ENTER vs dry-run PASS] path={dry_run_signals}",
                f"  offline first: sb={fe.get('session_bar_offline')} {fe.get('leg_offline')}",
                f"  stream first:  sb={fe.get('session_bar_stream')} {fe.get('leg_stream')}",
                f"  bar_delta={fe.get('bar_delta')} matched={fe.get('n_matched', 0)} "
                f"→ {'PASS' if fe.get('n_matched') == 1 or (fe.get('n_replay') == 0 and fe.get('n_live') == 0) else 'FAIL'}",
                f"  live_sim ENTER all: {ls.get('n_replay', 0)}  dry PASS: {ls.get('n_live', 0)}  "
                f"matched(±{tolerance_bars}): {ls.get('n_matched', 0)}",
            ]
        )

    if se_alpha_signals is not None and se_alpha_signals.exists():
        se_decision = load_se_alpha_decisions(
            se_alpha_signals,
            date=date,
            max_session_bar=max_session_bar,
        )
        report["se_alpha_path"] = str(se_alpha_signals)
        report["se_alpha_decision"] = se_decision.to_dict("records")
        report["replay_vs_se_alpha_decision"] = diff_signal_frames(
            decision_replay, se_decision, time_tolerance_bars=0
        )
        sd = report["replay_vs_se_alpha_decision"]["summary"]
        lines.extend(
            [
                "",
                f"[SE alpha decision vs replay decision] path={se_alpha_signals}",
                f"  se decision: {sd.get('n_live', 0)}  matched: {sd.get('n_matched', 0)}",
                f"  match rate: {sd.get('match_rate_replay', 0):.1%}",
            ]
        )

    # --- Exit lifecycle: offline LIVE_REPLAY EXIT vs fill_audit CLOSE ---
    audit_path = fill_audit_path or date_scoped_fill_audit_path(date)
    if not Path(audit_path).exists():
        from qqq_btc.live.fill_audit_writer import default_audit_path

        legacy = default_audit_path()
        if legacy.exists():
            audit_path = legacy
    offline_exits = collect_replay_exits(
        df,
        target_day=date,
        warmup_from_day=warmup_from_day,
        warmup_through_day=through,
        max_session_bar=max_session_bar,
    )
    live_exits = load_fill_audit_exits(audit_path, date, dedupe=True)
    if max_session_bar is not None and not live_exits.empty:
        live_exits = live_exits[
            pd.to_numeric(live_exits["session_bar"], errors="coerce") <= int(max_session_bar)
        ].copy()
    report["exit_lifecycle"] = diff_exit_lifecycle(
        offline_exits, live_exits, time_tolerance_bars=int(tolerance_bars)
    )
    report["first_exit"] = first_exit_diff(
        offline_exits, live_exits, time_tolerance_bars=int(tolerance_bars)
    )
    report["fill_model_audit"] = audit_fill_model_declared(audit_path, date)
    report["fill_audit_path"] = str(audit_path)
    report["offline_exits"] = offline_exits.to_dict("records")
    report["live_exits"] = live_exits.to_dict("records")
    fe_x = report["first_exit"]["summary"]
    el = report["exit_lifecycle"]["summary"]
    fm = report["fill_model_audit"]
    lines.extend(
        [
            "",
            f"[Exit lifecycle — LIVE_REPLAY EXIT vs fill_audit CLOSE] path={audit_path}",
            f"  offline exits: {el.get('n_replay', 0)}  live closes: {el.get('n_live', 0)}  "
            f"matched: {el.get('n_matched', 0)} → {'PASS' if el.get('pass') else 'FAIL'}",
            f"  first exit: offline sb={fe_x.get('session_bar_offline')} "
            f"{fe_x.get('leg_offline')} {fe_x.get('reason_offline')} | "
            f"stream sb={fe_x.get('session_bar_stream')} "
            f"{fe_x.get('leg_stream')} {fe_x.get('reason_stream')} | "
            f"delta={fe_x.get('bar_delta')} → {'PASS' if fe_x.get('pass') else 'FAIL'}",
            f"  fill model_frac median={fm.get('model_frac_median')} "
            f"(target={fm.get('target')}) realized_fill_median={fm.get('fill_spread_frac_median')} "
            f"→ {'PASS' if fm.get('pass') else ('SKIP' if fm.get('pass') is None else 'FAIL')}",
        ]
    )

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
    parser.add_argument("--se-alpha-signals", default=None, help="SE alpha frame audit CSV(可选)")
    parser.add_argument("--output", default=None, help="JSON 报告路径")
    parser.add_argument("--tolerance-bars", type=int, default=1)
    args = parser.parse_args()

    pq = Path(args.parquet).expanduser() if args.parquet else _default_parquet()
    if pq is None or not pq.exists():
        print("ERROR: --parquet 未指定或文件不存在", file=sys.stderr)
        sys.exit(1)

    dry = Path(args.dry_run_signals).expanduser() if args.dry_run_signals else None
    se_alpha = Path(args.se_alpha_signals).expanduser() if args.se_alpha_signals else None
    out = Path(args.output).expanduser() if args.output else None
    report = run_day_diff(
        parquet=pq,
        date=args.date,
        dry_run_signals=dry,
        se_alpha_signals=se_alpha,
        output=out,
        tolerance_bars=int(args.tolerance_bars),
    )
    ds = report["decision_replay_vs_live"]["summary"]
    decision_ok = ds.get("n_replay", 0) == ds.get("n_live", 0) == ds.get("n_matched", 0)
    sys.exit(0 if decision_ok else 2)


if __name__ == "__main__":
    main()
