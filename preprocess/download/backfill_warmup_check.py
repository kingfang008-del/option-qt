#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补数特征预热检查：窗口指标需要过去若干交易日，检测是否中断/缺数。

默认关注:
  - QQQ / VIXY 1min resampled（feature_merge / vix_level / gap / ATR）
  - 目标日前 N 个交易日是否连续存在
  - 单日 RTH 分钟条数是否明显不足
  - VIXY 更长历史月文件是否齐（vix_level 全局滚动）

用法:
  python preprocess/download/backfill_warmup_check.py \\
      --start-date 2026-07-14 --end-date 2026-07-14 \\
      --report /tmp/warmup_report.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO = _SCRIPT_DIR.parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from preprocess.download.dte_utils import _nyse_calendar  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("backfill_warmup_check")

NY = "America/New_York"
DEFAULT_STOCK_ROOT = Path.home() / "train_data/spnq_train_resampled"
DEFAULT_WARMUP_DAYS = 10
DEFAULT_VIX_HISTORY_MONTHS = 7
# RTH 09:30-16:00 ≈ 390 根 1min；低于此视为当日残缺
MIN_BARS_1MIN = 300


def nyse_trading_days(start: str, end: str) -> list[str]:
    cal = _nyse_calendar()
    sched = cal.schedule(start_date=start, end_date=end)
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in sched.index]


def prior_trading_days(anchor: str, n: int) -> list[str]:
    """返回 anchor 之前（不含当日）最近 n 个 NYSE 交易日。"""
    if n <= 0:
        return []
    # 日历缓冲：约 2×n + 周末/假日
    start = (pd.Timestamp(anchor) - pd.Timedelta(days=max(21, n * 4))).strftime("%Y-%m-%d")
    end = (pd.Timestamp(anchor) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    days = nyse_trading_days(start, end)
    return days[-n:] if len(days) >= n else days


def month_span_before(ym: str, n_months: int) -> list[str]:
    y, m = [int(x) for x in ym.split("-")]
    out: list[str] = []
    for i in range(n_months - 1, -1, -1):
        mm = m - i
        yy = y
        while mm <= 0:
            mm += 12
            yy -= 1
        out.append(f"{yy:04d}-{mm:02d}")
    return out


def _stock_1min_path(root: Path, symbol: str, ym: str) -> Path:
    return root / symbol / "regular/09:30-16:00/1min" / f"{ym}.parquet"


def days_and_bars_in_month(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    df = pd.read_parquet(path, columns=["timestamp"])
    if df.empty:
        return {}
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    try:
        ts = ts.dt.tz_convert(NY)
    except Exception:
        pass
    return ts.dt.strftime("%Y-%m-%d").value_counts().astype(int).to_dict()


def check_symbol_warmup(
    *,
    symbol: str,
    target_dates: list[str],
    stock_root: Path,
    warmup_trading_days: int,
    min_bars_1min: int = MIN_BARS_1MIN,
) -> dict[str, Any]:
    need_days: set[str] = set()
    for d in target_dates:
        need_days.update(prior_trading_days(d, warmup_trading_days))
        need_days.add(d)
    months = sorted({d[:7] for d in need_days})
    present_bars: dict[str, int] = {}
    missing_files: list[str] = []
    for ym in months:
        path = _stock_1min_path(stock_root, symbol, ym)
        if not path.is_file():
            missing_files.append(str(path))
            continue
        present_bars.update(days_and_bars_in_month(path))

    missing_days = sorted(d for d in need_days if d not in present_bars)
    thin_days = sorted(
        d for d, n in present_bars.items() if d in need_days and int(n) < min_bars_1min
    )
    # 连续性：对每个 target，检查 prior window 是否有缺口
    interrupted: list[dict[str, Any]] = []
    for d in target_dates:
        prior = prior_trading_days(d, warmup_trading_days)
        gaps = [x for x in prior if x not in present_bars or present_bars.get(x, 0) < min_bars_1min]
        if gaps:
            interrupted.append(
                {
                    "target_date": d,
                    "warmup_need": warmup_trading_days,
                    "warmup_have": warmup_trading_days - len(gaps),
                    "gap_days": gaps,
                }
            )

    ok = not missing_files and not missing_days and not interrupted
    return {
        "symbol": symbol,
        "ok": ok,
        "warmup_trading_days": warmup_trading_days,
        "needed_days": sorted(need_days),
        "missing_month_files": missing_files,
        "missing_days": missing_days,
        "thin_days": thin_days,
        "interrupted_targets": interrupted,
        "present_day_count": len([d for d in need_days if d in present_bars]),
        "needed_day_count": len(need_days),
    }


def check_vix_history(
    *,
    start_date: str,
    stock_root: Path,
    history_months: int,
) -> dict[str, Any]:
    ym = start_date[:7]
    months = month_span_before(ym, history_months)
    missing = []
    present = []
    for m in months:
        path = _stock_1min_path(stock_root, "VIXY", m)
        if path.is_file():
            present.append(m)
        else:
            missing.append(str(path))
    return {
        "symbol": "VIXY",
        "ok": not missing,
        "history_months": history_months,
        "required_months": months,
        "present_months": present,
        "missing_month_files": missing,
        "note": "vix_level 全局滚动通常需要数月历史；缺月会导致 put_gate/vix 特征塌缩",
    }


def run_warmup_check(
    *,
    start_date: str,
    end_date: str,
    symbols: list[str],
    stock_root: Path,
    warmup_trading_days: int = DEFAULT_WARMUP_DAYS,
    vix_history_months: int = DEFAULT_VIX_HISTORY_MONTHS,
    min_bars_1min: int = MIN_BARS_1MIN,
) -> dict[str, Any]:
    targets = nyse_trading_days(start_date, end_date)
    if not targets:
        # 若日历无结果，仍用用户给出的端点做检查锚点
        targets = sorted({start_date, end_date})

    per_symbol = [
        check_symbol_warmup(
            symbol=sym,
            target_dates=targets,
            stock_root=stock_root,
            warmup_trading_days=warmup_trading_days,
            min_bars_1min=min_bars_1min,
        )
        for sym in symbols
    ]
    # 特征侧总是需要 VIXY，即使 symbols 只有 QQQ
    if "VIXY" not in {s.upper() for s in symbols}:
        per_symbol.append(
            check_symbol_warmup(
                symbol="VIXY",
                target_dates=targets,
                stock_root=stock_root,
                warmup_trading_days=warmup_trading_days,
                min_bars_1min=min_bars_1min,
            )
        )
    vix_hist = check_vix_history(
        start_date=start_date,
        stock_root=stock_root,
        history_months=vix_history_months,
    )

    blockers: list[str] = []
    warnings: list[str] = []
    for row in per_symbol:
        if row["missing_month_files"]:
            blockers.append(f"{row['symbol']}: missing month files {row['missing_month_files']}")
        if row["interrupted_targets"]:
            for it in row["interrupted_targets"]:
                blockers.append(
                    f"{row['symbol']}: {it['target_date']} warmup interrupted gaps={it['gap_days']}"
                )
        if row["thin_days"]:
            warnings.append(f"{row['symbol']}: thin RTH bars on {row['thin_days']}")
    if not vix_hist["ok"]:
        blockers.append(f"VIXY history months missing: {vix_hist['missing_month_files']}")

    coverage = coverage_vs_today(
        symbols=list({s.upper() for s in symbols} | {"VIXY"}),
        stock_root=stock_root,
        asof=None,
        lookback_calendar_days=45,
        min_bars_1min=min_bars_1min,
    )
    report = {
        "ok": len(blockers) == 0,
        "start_date": start_date,
        "end_date": end_date,
        "target_dates": targets,
        "warmup_trading_days": warmup_trading_days,
        "vix_history_months": vix_history_months,
        "min_bars_1min": min_bars_1min,
        "stock_root": str(stock_root),
        "symbols": per_symbol,
        "vix_history": vix_hist,
        "coverage_vs_today": coverage,
        "blockers": blockers,
        "warnings": warnings,
        "hint": (
            "窗口特征（EMA/ATR/gap/vix_level）依赖过去交易日连续分钟数据。"
            "若 blockers 非空：先补齐 spnq_train_resampled 对应日/月，再生成特征；"
            "生成后默认 rolling_norm（经典离线）；流式对齐再选 frozen_norm。"
            "coverage_vs_today：相对今天缺哪些交易日（数据落后）。"
        ),
    }
    return report


def coverage_vs_today(
    *,
    symbols: list[str] | None = None,
    stock_root: Path | None = None,
    asof: str | None = None,
    lookback_calendar_days: int = 45,
    min_bars_1min: int = MIN_BARS_1MIN,
) -> dict[str, Any]:
    """相对“当前/asof”看最近一段交易日还缺哪些天（数据落后到哪）。"""
    root = Path(stock_root or DEFAULT_STOCK_ROOT).expanduser()
    syms = [s.upper() for s in (symbols or ["QQQ", "VIXY"])]
    asof_day = asof or pd.Timestamp.now(tz=NY).strftime("%Y-%m-%d")
    start = (pd.Timestamp(asof_day) - pd.Timedelta(days=lookback_calendar_days)).strftime("%Y-%m-%d")
    # 含 asof 当天：若今天尚未收盘/未落盘，会显示为 missing（符合“相对当前缺数”）
    expected = nyse_trading_days(start, asof_day)
    per_symbol: list[dict[str, Any]] = []
    for sym in syms:
        months = sorted({d[:7] for d in expected})
        present_bars: dict[str, int] = {}
        for ym in months:
            present_bars.update(days_and_bars_in_month(_stock_1min_path(root, sym, ym)))
        present_ok = sorted(d for d in expected if present_bars.get(d, 0) >= min_bars_1min)
        missing = sorted(d for d in expected if d not in present_ok)
        latest = present_ok[-1] if present_ok else None
        lag_days = len([d for d in expected if d > latest]) if latest else len(expected)
        per_symbol.append(
            {
                "symbol": sym,
                "latest_present": latest,
                "lag_trading_days": lag_days,
                "missing_days": missing,
                "missing_count": len(missing),
                "expected_count": len(expected),
                "present_count": len(present_ok),
            }
        )
    all_missing = sorted({d for row in per_symbol for d in row["missing_days"]})
    return {
        "asof": asof_day,
        "lookback_calendar_days": lookback_calendar_days,
        "expected_trading_days": expected,
        "symbols": per_symbol,
        "union_missing_days": all_missing,
        "ok": len(all_missing) == 0,
        "hint": "missing_days = 相对 asof 的近期交易日中，1min 数据缺失或过薄的日期",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start-date", default="", help="与 --end-date 一起做目标区间预热检查")
    p.add_argument("--end-date", default="")
    p.add_argument("--symbols", default="QQQ")
    p.add_argument("--stock-root", default=str(DEFAULT_STOCK_ROOT))
    p.add_argument("--warmup-trading-days", type=int, default=DEFAULT_WARMUP_DAYS)
    p.add_argument("--vix-history-months", type=int, default=DEFAULT_VIX_HISTORY_MONTHS)
    p.add_argument("--min-bars-1min", type=int, default=MIN_BARS_1MIN)
    p.add_argument("--report", default="")
    p.add_argument("--strict", action="store_true", help="有 blockers 时 exit 2")
    p.add_argument(
        "--coverage-only",
        action="store_true",
        help="只输出相对今天的缺数日历（不要求 --start-date/--end-date）",
    )
    p.add_argument("--asof", default="", help="coverage 基准日，默认今天 NY")
    p.add_argument("--lookback-calendar-days", type=int, default=45)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    stock_root = Path(args.stock_root).expanduser()
    if args.coverage_only:
        report: dict[str, Any] = coverage_vs_today(
            symbols=list(set(symbols) | {"VIXY"}),
            stock_root=stock_root,
            asof=args.asof or None,
            lookback_calendar_days=int(args.lookback_calendar_days),
            min_bars_1min=int(args.min_bars_1min),
        )
        report["ok"] = bool(report.get("ok"))
        report["blockers"] = (
            [f"coverage lag: {report['union_missing_days']}"]
            if not report["ok"]
            else []
        )
        report["warnings"] = []
    else:
        if not args.start_date or not args.end_date:
            logger.error("需要 --start-date/--end-date，或改用 --coverage-only")
            return 2
        report = run_warmup_check(
            start_date=args.start_date,
            end_date=args.end_date,
            symbols=symbols,
            stock_root=stock_root,
            warmup_trading_days=int(args.warmup_trading_days),
            vix_history_months=int(args.vix_history_months),
            min_bars_1min=int(args.min_bars_1min),
        )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        out = Path(args.report).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        logger.info("wrote %s ok=%s blockers=%d", out, report.get("ok"), len(report.get("blockers") or []))
    else:
        print(text)
    for b in report.get("blockers") or []:
        logger.error("BLOCKER: %s", b)
    for w in report.get("warnings") or []:
        logger.warning("WARN: %s", w)
    if args.strict and not report.get("ok"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
