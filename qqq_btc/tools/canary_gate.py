#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
开盘前 canary 门禁 —— 用昨日数据把 live 栈(FCS→SE→OMS)与 strict replay 对拍,
写 gate JSON;OMS 侧 entry_guard 读到不通过/过期即当日禁止新开仓。

流程:
  1. 调 compare_stream_replay_day.py --date <上一交易日>(全速 sync 发球)
  2. 读 verdict JSON 的 replay vs stream 匹配率
  3. match_rate >= --min-match-rate → trading_allowed=true,否则 false
  4. 写 ~/quant_project/shadow/canary_gate.json(entry_guard 默认路径)

用法(建议 cron 开盘前, 如 08:00 ET):
  python qqq_btc/tools/canary_gate.py                # 自动取上一交易日
  python qqq_btc/tools/canary_gate.py --date 2026-06-26
  python qqq_btc/tools/canary_gate.py --date 2026-06-26 --skip-redis   # 复用已有 signals
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from pytz import timezone

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_NY = timezone("America/New_York")
_WORK = Path("/tmp/qqq_btc_stream_parity")
DEFAULT_GATE = Path.home() / "quant_project" / "shadow" / "canary_gate.json"


def previous_trading_day(ref: datetime | None = None) -> str:
    d = (ref or datetime.now(tz=_NY)).date() - timedelta(days=1)
    while d.weekday() >= 5:  # Sat/Sun
        d -= timedelta(days=1)
    return d.isoformat()


def next_close_epoch(after: datetime) -> float:
    """gate 有效期: 下一个交易日 16:00 ET。"""
    d = after.date()
    while True:
        close = _NY.localize(datetime(d.year, d.month, d.day, 16, 0))
        if close > after and d.weekday() < 5:
            return close.timestamp()
        d += timedelta(days=1)


def write_gate(
    gate_path: Path,
    *,
    allowed: bool,
    date: str,
    match_rate: float,
    min_match_rate: float,
    detail: dict,
    reason: str = "",
) -> None:
    now = datetime.now(tz=_NY)
    payload = {
        "trading_allowed": bool(allowed),
        "canary_date": date,
        "match_rate": round(float(match_rate), 4),
        "min_match_rate": float(min_match_rate),
        "reason": reason,
        "generated_at": now.isoformat(),
        "expires_at": next_close_epoch(now),
        "detail": detail,
    }
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    flag = "✅ ALLOWED" if allowed else "🚫 BLOCKED"
    print(f"[canary_gate] {flag} match_rate={match_rate:.1%} (min {min_match_rate:.0%}) → {gate_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="开盘前 canary parity 门禁")
    parser.add_argument("--date", default=None, help="canary 日 YYYY-MM-DD(默认上一交易日)")
    parser.add_argument("--min-match-rate", type=float, default=0.90)
    parser.add_argument("--gate-path", default=str(DEFAULT_GATE))
    parser.add_argument("--warmup-from", default="same-day")
    parser.add_argument("--skip-redis", action="store_true", help="复用已有 signals CSV")
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    date = args.date or previous_trading_day()
    gate_path = Path(args.gate_path).expanduser()
    verdict_path = _WORK / f"diff_{date}_verdict.json"

    cmd = [
        sys.executable,
        str(_REPO / "qqq_btc" / "tools" / "compare_stream_replay_day.py"),
        "--date",
        date,
        "--warmup-from",
        args.warmup_from,
        "--output",
        str(_WORK / f"diff_{date}.json"),
    ]
    if args.skip_redis:
        cmd.append("--skip-redis")
    if args.skip_infer:
        cmd.append("--skip-infer")
    if args.parquet:
        cmd.extend(["--parquet", args.parquet])
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])

    print("[canary_gate]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(_REPO))

    if not verdict_path.exists():
        write_gate(
            gate_path,
            allowed=False,
            date=date,
            match_rate=0.0,
            min_match_rate=args.min_match_rate,
            detail={"error": f"verdict missing (compare exit={proc.returncode})"},
            reason="canary_run_failed",
        )
        return 2

    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    summary = verdict.get("replay_vs_dry_run_summary", {}) or {}
    n_replay = int(summary.get("n_replay", 0) or 0)
    n_live = int(summary.get("n_live", 0) or 0)
    n_matched = int(summary.get("n_matched", 0) or 0)

    if n_replay == 0 and n_live == 0:
        # 双方都无信号: 不能当成对拍通过(样本不足),禁止开仓
        match_rate = 0.0
        reason = "no_signal_day_insufficient"
        allowed = False
    elif n_replay == 0:
        # replay 无信号但 live 有 → live 侧膨胀, 危险
        match_rate = 0.0
        reason = f"live_only_signals n_live={n_live}"
        allowed = False
    else:
        # 双向惩罚: live 漏报和多报都要压 match rate
        match_rate = n_matched / max(n_replay, n_live)
        reason = ""
        allowed = match_rate >= args.min_match_rate

    write_gate(
        gate_path,
        allowed=allowed,
        date=date,
        match_rate=match_rate,
        min_match_rate=args.min_match_rate,
        detail={
            "n_replay": n_replay,
            "n_live": n_live,
            "n_matched": n_matched,
            "verdict_json": str(verdict_path),
            "decision_replay_vs_live_ok": verdict.get("decision_replay_vs_live_ok"),
        },
        reason=reason,
    )
    return 0 if allowed else 3


if __name__ == "__main__":
    raise SystemExit(main())
