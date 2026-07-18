#!/usr/bin/env python3
"""股价分钟 bar 标签约定（W1 对拍固化）。

Massive / Polygon aggregates 原始是**左标签**（``09:30`` = ``[09:30,09:31)``）。
本项目 W1 / V4 特征、期权 1m、离线 infer、实盘 ``+60s`` 桥接统一要求
**右标签**（``09:31`` = bar 收盘时刻）。

约定:
  - ``spnq_train``（raw）保持左标签，勿改
  - ``spnq_train_resampled`` 进入 feature_merge 前必须是右标签
  - 检测/纠正只改 resampled，不碰 raw

用法:
  python -m qqq_btc.common.bar_label_convention --scan
  python -m qqq_btc.common.bar_label_convention --fix --symbols QQQ,VIXY \\
      --start 2026-07-01 --end 2026-07-13
"""
from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

TZ = "America/New_York"
DEFAULT_STOCK_RESAMP = Path.home() / "train_data/spnq_train_resampled"
DEFAULT_SYMBOLS = ("QQQ", "VIXY")
DEFAULT_RES = ("1min", "5min")
# W1 / offline feature convention
EXPECTED_FIRST_RTH = "09:31"
LEFT_FIRST_RTH = "09:30"


def _to_ny(ts: pd.Series) -> pd.Series:
    out = pd.to_datetime(ts)
    if getattr(out.dt, "tz", None) is None:
        return out.dt.tz_localize(TZ)
    return out.dt.tz_convert(TZ)


def month_span(start: str, end: str) -> list[str]:
    return [
        p.strftime("%Y-%m")
        for p in pd.period_range(str(start)[:7], str(end)[:7], freq="M")
    ]


def stock_month_path(
    root: Path,
    symbol: str,
    ym: str,
    *,
    res: str = "1min",
) -> Path:
    return (
        Path(root).expanduser()
        / symbol
        / "regular"
        / "09:30-16:00"
        / res
        / f"{ym}.parquet"
    )


@dataclass
class BarLabelFileReport:
    symbol: str
    ym: str
    res: str
    path: str
    exists: bool
    label: str = "unknown"  # right | left | mixed | empty | missing | unknown
    first_ts: str | None = None
    first_hhmm: str | None = None
    n_rows: int = 0
    n_days: int = 0
    n_left_days: int = 0
    n_right_days: int = 0
    ok_for_w1: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_parquet_label(path: Path) -> BarLabelFileReport:
    """检查单月 parquet 是左/右标签（按每个交易日首根 HH:MM）。"""
    symbol = "?"
    ym = path.stem
    res = path.parent.name
    try:
        # .../SYM/regular/09:30-16:00/1min/YYYY-MM.parquet
        symbol = path.parents[3].name
    except Exception:
        pass
    base = BarLabelFileReport(
        symbol=symbol,
        ym=ym,
        res=res,
        path=str(path),
        exists=path.is_file(),
    )
    if not path.is_file():
        base.label = "missing"
        base.note = "file missing"
        return base
    try:
        df = pd.read_parquet(path, columns=["timestamp"])
    except Exception as exc:
        base.label = "unknown"
        base.note = f"read failed: {exc}"
        return base
    if df.empty or "timestamp" not in df.columns:
        base.label = "empty"
        base.note = "empty parquet"
        return base
    ts = _to_ny(df["timestamp"]).sort_values()
    base.n_rows = int(len(ts))
    base.first_ts = str(ts.iloc[0])
    base.first_hhmm = ts.iloc[0].strftime("%H:%M")
    days = ts.dt.strftime("%Y-%m-%d")
    first_by_day = ts.groupby(days).first()
    base.n_days = int(len(first_by_day))
    hhmm = first_by_day.dt.strftime("%H:%M")
    base.n_left_days = int((hhmm == LEFT_FIRST_RTH).sum())
    base.n_right_days = int((hhmm == EXPECTED_FIRST_RTH).sum())
    if base.n_left_days and base.n_right_days:
        base.label = "mixed"
        base.note = (
            f"left_days={base.n_left_days} right_days={base.n_right_days}; "
            "W1 要求全部右标签"
        )
        base.ok_for_w1 = False
    elif base.n_left_days:
        base.label = "left"
        base.note = (
            f"Massive 左标签（首根 {LEFT_FIRST_RTH}）；"
            "进特征前需 +1min → 右标签"
        )
        base.ok_for_w1 = False
    elif base.n_right_days:
        base.label = "right"
        base.note = f"已是 W1 右标签（首根 {EXPECTED_FIRST_RTH}）"
        base.ok_for_w1 = True
    else:
        base.label = "unknown"
        base.note = f"首根既非 {LEFT_FIRST_RTH} 也非 {EXPECTED_FIRST_RTH}: {hhmm.value_counts().to_dict()}"
        base.ok_for_w1 = False
    return base


def scan_bar_labels(
    *,
    stock_root: Path | None = None,
    symbols: Iterable[str] = DEFAULT_SYMBOLS,
    start: str | None = None,
    end: str | None = None,
    months: Iterable[str] | None = None,
    resolutions: Iterable[str] = DEFAULT_RES,
) -> dict[str, Any]:
    root = Path(stock_root or DEFAULT_STOCK_RESAMP).expanduser()
    syms = [s.strip().upper() for s in symbols if str(s).strip()]
    if months is None:
        if start and end:
            yms = month_span(start, end)
        else:
            # 默认扫各符号 1min 目录下已有月份
            yms = []
            for sym in syms:
                d = root / sym / "regular/09:30-16:00/1min"
                if d.is_dir():
                    yms.extend(p.stem for p in d.glob("????-??.parquet"))
            yms = sorted(set(yms))
    else:
        yms = list(months)
    files: list[BarLabelFileReport] = []
    for sym in syms:
        for ym in yms:
            for res in resolutions:
                files.append(
                    inspect_parquet_label(stock_month_path(root, sym, ym, res=res))
                )
    bad = [f for f in files if f.exists and not f.ok_for_w1]
    missing = [f for f in files if not f.exists]
    return {
        "convention": {
            "name": "W1_parity_right_label",
            "expected_first_rth": EXPECTED_FIRST_RTH,
            "massive_raw": "left_label (keep in spnq_train)",
            "resampled_required": "right_label",
            "live_bridge": "FCS alpha_label_ts + 60s → offline right stamp",
        },
        "stock_root": str(root),
        "symbols": syms,
        "months": yms,
        "ok": len(bad) == 0,
        "n_files": len(files),
        "n_ok": sum(1 for f in files if f.ok_for_w1),
        "n_bad": len(bad),
        "n_missing": len(missing),
        "files": [f.to_dict() for f in files],
        "bad_files": [f.to_dict() for f in bad],
    }


def fix_parquet_to_right_label(
    path: Path,
    *,
    dry_run: bool = False,
    backup: bool = True,
) -> dict[str, Any]:
    """将单月文件从左标签改为右标签（timestamp +1min，NY tz）。"""
    path = Path(path).expanduser()
    report = inspect_parquet_label(path)
    out: dict[str, Any] = {
        "path": str(path),
        "before": report.to_dict(),
        "dry_run": dry_run,
        "changed": False,
    }
    if not report.exists:
        out["status"] = "missing"
        return out
    if report.ok_for_w1 and report.label == "right":
        out["status"] = "already_right"
        return out
    if report.label not in {"left", "mixed"}:
        out["status"] = f"skip_{report.label}"
        out["note"] = report.note
        return out

    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        out["status"] = "no_timestamp"
        return out
    df = df.copy()
    df["timestamp"] = _to_ny(df["timestamp"])
    # mixed: only shift days that are still left-labeled
    if report.label == "mixed":
        day = df["timestamp"].dt.strftime("%Y-%m-%d")
        first_hhmm = df.groupby(day)["timestamp"].min().dt.strftime("%H:%M")
        need_days = set(first_hhmm[first_hhmm == LEFT_FIRST_RTH].index)
        need = day.isin(need_days)
        df.loc[need, "timestamp"] = df.loc[need, "timestamp"] + pd.Timedelta(
            minutes=1
        )
    else:
        df["timestamp"] = df["timestamp"] + pd.Timedelta(minutes=1)
    df["timestamp"] = _to_ny(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    df = df.reset_index(drop=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = Path(str(path) + f".bak_left_label_{stamp}")
    out["backup"] = str(bak)
    if dry_run:
        # validate without write
        tmp_first = df["timestamp"].iloc[0].strftime("%H:%M")
        out["status"] = "dry_run"
        out["would_first_hhmm"] = tmp_first
        out["changed"] = True
        return out

    if backup and not bak.exists():
        shutil.copy2(path, bak)
    df.to_parquet(path, index=False)
    after = inspect_parquet_label(path)
    out["after"] = after.to_dict()
    out["changed"] = True
    out["status"] = "fixed" if after.ok_for_w1 else "fixed_but_still_bad"
    return out


def fix_bar_labels(
    *,
    stock_root: Path | None = None,
    symbols: Iterable[str] = DEFAULT_SYMBOLS,
    start: str | None = None,
    end: str | None = None,
    months: Iterable[str] | None = None,
    resolutions: Iterable[str] = DEFAULT_RES,
    dry_run: bool = False,
    only_bad: bool = True,
) -> dict[str, Any]:
    scan = scan_bar_labels(
        stock_root=stock_root,
        symbols=symbols,
        start=start,
        end=end,
        months=months,
        resolutions=resolutions,
    )
    targets = scan["bad_files"] if only_bad else scan["files"]
    results = []
    for row in targets:
        if not row.get("exists"):
            continue
        results.append(
            fix_parquet_to_right_label(Path(row["path"]), dry_run=dry_run)
        )
    rescan = scan_bar_labels(
        stock_root=stock_root,
        symbols=symbols,
        start=start,
        end=end,
        months=months or scan["months"],
        resolutions=resolutions,
    )
    return {
        "dry_run": dry_run,
        "n_attempted": len(results),
        "n_changed": sum(1 for r in results if r.get("changed")),
        "results": results,
        "before_ok": scan["ok"],
        "after_ok": rescan["ok"],
        "scan_after": rescan,
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stock-root", type=Path, default=DEFAULT_STOCK_RESAMP)
    ap.add_argument("--symbols", default="QQQ,VIXY")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--res", default="1min,5min")
    args = ap.parse_args(argv)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    resolutions = [r.strip() for r in args.res.split(",") if r.strip()]
    if args.fix:
        out = fix_bar_labels(
            stock_root=args.stock_root,
            symbols=symbols,
            start=args.start,
            end=args.end,
            resolutions=resolutions,
            dry_run=args.dry_run,
        )
        print(
            f"fix dry_run={out['dry_run']} changed={out['n_changed']}/{out['n_attempted']} "
            f"after_ok={out['after_ok']}"
        )
        for r in out["results"]:
            print(" ", r.get("status"), r.get("path"))
        return 0 if out["after_ok"] or args.dry_run else 1
    # default scan
    scan = scan_bar_labels(
        stock_root=args.stock_root,
        symbols=symbols,
        start=args.start,
        end=args.end,
        resolutions=resolutions,
    )
    print(
        f"scan ok={scan['ok']} bad={scan['n_bad']} missing={scan['n_missing']} "
        f"root={scan['stock_root']}"
    )
    for f in scan["files"]:
        if not f["exists"]:
            continue
        mark = "OK" if f["ok_for_w1"] else "BAD"
        print(
            f"  [{mark}] {f['symbol']} {f['ym']} {f['res']} {f['label']} "
            f"first={f['first_hhmm']} {f['note']}"
        )
    return 0 if scan["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
