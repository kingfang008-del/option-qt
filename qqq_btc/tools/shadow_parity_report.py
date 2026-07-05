#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dry-run / 影子模式对账报告 —— 汇总 fill_audit 与 replay 导出 CSV 的 parity。

用法(实盘 dry-run 一天后):
  cd New_Pro/baseline_qqq
  set -a && source config/minimal_stack.env && set +a
  # RUN_MODE=REALTIME_DRY, TRADING_ENABLED=0 下成对启动:
  #   python ../../qqq_btc/tools/run_live_signal_qqq.py
  #   python ../../qqq_btc/tools/run_live_exec_qqq.py

  python ../../qqq_btc/tools/shadow_parity_report.py \\
    --audit-log ~/quant_project/shadow/fill_audit.csv \\
    --replay-trades ../../qqq_btc/results/replay_trades_q2_best.csv

环境变量(可选):
  QQQ_BTC_FILL_AUDIT_PATH  fill 审计 CSV 路径
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qqq_btc.live.fill_audit_writer import default_audit_path
from qqq_btc.qqq import config as qcfg
from qqq_btc.tools.parity_audit import audit_exit_reasons, audit_fill


def _default_replay_trades() -> Path | None:
    p = _REPO / "qqq_btc" / "results" / "replay_trades_q2_best.csv"
    return p if p.exists() else None


def run_report(
    *,
    audit_log: Path,
    replay_trades: Path | None,
    target_frac: float,
    output: Path | None,
) -> dict:
    fill = audit_fill(audit_log, target_frac=target_frac)
    exits = audit_exit_reasons(audit_log, replay_trades)
    report = {
        "audit_log": str(audit_log),
        "replay_trades": str(replay_trades) if replay_trades else None,
        "fill": fill,
        "exits": exits,
        "replay_config": {
            "entry_quantile": qcfg.REPLAY.entry_quantile,
            "max_trades_per_day": qcfg.REPLAY.max_trades_per_day,
            "daily_loss_stop": qcfg.REPLAY.daily_loss_stop,
            "vol_scale_ref": qcfg.EXIT_RAILS.vol_scale_ref,
        },
    }
    ok_fill = bool(fill.get("pass"))
    ok_exit = bool(exits.get("pass", exits.get("n_close", 0) > 0))
    report["overall_pass"] = ok_fill and ok_exit

    lines = [
        "=== qqq_btc Shadow Parity Report ===",
        f"audit_log: {audit_log}",
        "",
        "[Fill vs model 0.775]",
        f"  n={fill.get('n', 0)} median={fill.get('median', 'n/a')} target={fill.get('target', target_frac)}",
        f"  pass={fill.get('pass', False)}",
        "",
        "[Exit reason distribution]",
        f"  closes={exits.get('n_close', 0)}",
    ]
    if "live_distribution" in exits:
        for k, v in sorted(exits["live_distribution"].items(), key=lambda x: -x[1])[:8]:
            lines.append(f"    {k}: {v:.1%}")
    if "distribution_l1" in exits:
        lines.append(f"  L1 vs replay={exits['distribution_l1']:.3f} pass={exits.get('pass')}")
    lines.append("")
    lines.append(f"overall_pass={report['overall_pass']}")

    text = "\n".join(lines)
    print(text)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="qqq_btc dry-run shadow parity 汇总")
    parser.add_argument(
        "--audit-log",
        default=str(default_audit_path()),
        help="fill_audit.csv(QQQ_BTC_FILL_AUDIT_PATH)",
    )
    parser.add_argument(
        "--replay-trades",
        default=None,
        help="strict replay 导出的 trades CSV(含 exit_reason)",
    )
    parser.add_argument("--output", default=None, help="JSON 报告路径")
    parser.add_argument(
        "--target-frac",
        type=float,
        default=qcfg.FILL_MODEL.entry_frac,
    )
    args = parser.parse_args()

    audit = Path(args.audit_log).expanduser()
    if not audit.exists():
        print(f"ERROR: audit log not found: {audit}", file=sys.stderr)
        print("Dry-run 需先启动 run_live_*_qqq.py 并产生成交审计。", file=sys.stderr)
        sys.exit(1)

    replay = Path(args.replay_trades).expanduser() if args.replay_trades else _default_replay_trades()
    out = Path(args.output).expanduser() if args.output else None
    report = run_report(
        audit_log=audit,
        replay_trades=replay,
        target_frac=float(args.target_frac),
        output=out,
    )
    sys.exit(0 if report.get("overall_pass") else 2)


if __name__ == "__main__":
    main()
