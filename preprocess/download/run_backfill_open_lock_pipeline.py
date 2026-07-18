#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""开盘价锁 4 合约 → quote 下载 → 1m/day_iv → 按日特征目录。

替代「先下全市场交易再提合约」的旧补数路径。

目录布局（与 step2_polygon_second_sniper_v1 一致）:
  $EXP/   默认 /mnt/s990/data/raw_1s/dte1_options_old_lock
    {SYM}/{SYM}_{date}.parquet          # 1s quote（历史已有 QQQ/）
    locked_targets_map_open_4bucket.parquet
    options_1m/{SYM}/{SYM}_{date}.parquet
    quote_options_day_iv/{SYM}/...
    by_date/{date}/
      lock_map.parquet
      manifest.json
      raw_1s.parquet          (symlink/copy)
      options_1m.parquet
      day_iv.parquet          (若算出来)
      features_1min.parquet   (--features 时；Dashboard Download 页不跑)

股价缺数/预热比对:
  ~/train_data/spnq_train_resampled/{QQQ,VIXY}/.../1min

用法:
  export MASSIVE_API_KEY=...
  python preprocess/download/run_backfill_open_lock_pipeline.py \\
      --start-date 2026-07-14 --end-date 2026-07-14

  # 只锁约预览
  python preprocess/download/run_backfill_open_lock_pipeline.py \\
      --start-date 2026-07-14 --end-date 2026-07-14 --lock-only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PREPROCESS_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _PREPROCESS_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("backfill_open_lock")

DEFAULT_CONFIG = _PREPROCESS_ROOT / "CONFIG" / "anchor_qqq_1dte_4bucket.json"
DEFAULT_EXP = Path("/mnt/s990/data/raw_1s/dte1_options_old_lock")
DEFAULT_STOCK_RESAMP = Path.home() / "train_data/spnq_train_resampled"
# 特征/归一化默认写到 train_data，避免混进 raw_1s quote 目录
DEFAULT_FEATURES_ROOT = Path.home() / "train_data/dte1_options_old_lock_feat"
DEFAULT_FEAT_HISTORY = Path.home() / "train_data/quote_features_raw"
DEFAULT_FROZEN_NORM = _REPO_ROOT / "qqq_btc/CONFIG/frozen_norm_qqq_daily.npz"
DEFAULT_PY = os.environ.get(
    "PYTHON",
    str(Path.home() / "anaconda3/envs/ibkr/bin/python"),
)


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    logger.info("$ %s", " ".join(cmd))
    merged = os.environ.copy()
    if env:
        merged.update(env)
    merged["PYTHONPATH"] = (
        str(_REPO_ROOT) + (os.pathsep + merged["PYTHONPATH"] if merged.get("PYTHONPATH") else "")
    )
    subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT), env=merged)


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def stage_by_date(
    exp: Path,
    map_path: Path,
    *,
    symbol: str,
    dates: list[str],
    features_root: Path | None = None,
) -> list[dict[str, Any]]:
    staged: list[dict[str, Any]] = []
    full_map = pd.read_parquet(map_path)
    full_map["date_str"] = full_map["date_str"].astype(str)
    feat_root = Path(features_root or exp).expanduser()
    for d in dates:
        day_dir = exp / "by_date" / d
        day_dir.mkdir(parents=True, exist_ok=True)
        day_map = full_map[full_map["date_str"] == d].copy()
        day_map_path = day_dir / "lock_map.parquet"
        day_map.to_parquet(day_map_path, index=False)

        raw_1s = exp / symbol / f"{symbol}_{d}.parquet"
        # 兼容旧 nested 布局 exp/raw_1s/{SYM}/...
        if not raw_1s.is_file():
            nested = exp / "raw_1s" / symbol / f"{symbol}_{d}.parquet"
            if nested.is_file():
                raw_1s = nested
        opt_1m = exp / "options_1m" / symbol / f"{symbol}_{d}.parquet"
        if raw_1s.is_file():
            _link_or_copy(raw_1s, day_dir / "raw_1s.parquet")
        if opt_1m.is_file():
            _link_or_copy(opt_1m, day_dir / "options_1m.parquet")

        day_iv_hits = sorted((exp / "quote_options_day_iv" / symbol).glob(f"**/{symbol}_{d}.parquet"))
        if not day_iv_hits:
            day_iv_hits = sorted((exp / "quote_options_day_iv" / symbol).glob(f"**/*{d}*.parquet"))
        if day_iv_hits:
            _link_or_copy(day_iv_hits[0], day_dir / "day_iv.parquet")

        feat_1m = feat_root / "quote_features_raw" / symbol / "regular/09:30-16:00/1min"
        # monthly parquet may contain many days; extract single day if present
        ym = d[:7]
        monthly = feat_1m / f"{ym}.parquet"
        feat_out = day_dir / "features_1min.parquet"
        if monthly.is_file():
            df = pd.read_parquet(monthly)
            if "timestamp" in df.columns:
                ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                try:
                    ts = ts.dt.tz_convert("America/New_York")
                except Exception:
                    pass
                mask = ts.dt.strftime("%Y-%m-%d") == d
                day_feat = df.loc[mask].copy()
                if not day_feat.empty:
                    day_feat.to_parquet(feat_out, index=False)

        feat_test_monthly = (
            feat_root / "quote_features_test" / symbol / "regular/09:30-16:00/1min" / f"{ym}.parquet"
        )
        feat_test_out = day_dir / "features_1min_norm.parquet"
        if feat_test_monthly.is_file():
            df = pd.read_parquet(feat_test_monthly)
            if "timestamp" in df.columns:
                ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                try:
                    ts = ts.dt.tz_convert("America/New_York")
                except Exception:
                    pass
                mask = ts.dt.strftime("%Y-%m-%d") == d
                day_feat = df.loc[mask].copy()
                if not day_feat.empty:
                    day_feat.to_parquet(feat_test_out, index=False)

        manifest = {
            "date": d,
            "symbol": symbol,
            "lock_map": str(day_map_path),
            "n_contracts": int(len(day_map)),
            "contracts": day_map["contract_symbol"].astype(str).tolist() if not day_map.empty else [],
            "stock_open": float(day_map["stock_open"].iloc[0]) if "stock_open" in day_map.columns and not day_map.empty else None,
            "raw_1s": str(raw_1s) if raw_1s.is_file() else None,
            "options_1m": str(opt_1m) if opt_1m.is_file() else None,
            "day_iv": str(day_dir / "day_iv.parquet") if (day_dir / "day_iv.parquet").exists() else None,
            "features_1min": str(feat_out) if feat_out.is_file() else None,
            "features_1min_norm": str(feat_test_out) if feat_test_out.is_file() else None,
            "features_root": str(feat_root),
            "lock_mode": "open_price_4bucket",
        }
        (day_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        staged.append(manifest)
        logger.info("staged by_date/%s contracts=%d", d, manifest["n_contracts"])
    return staged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--symbols", default="QQQ")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument(
        "--exp",
        default=str(DEFAULT_EXP),
        help="1s quote 根目录（默认 dte1_options_old_lock；文件写 {SYM}/{SYM}_{date}.parquet）",
    )
    p.add_argument("--python", default=DEFAULT_PY)
    p.add_argument("--assume-iv", type=float, default=0.22)
    p.add_argument("--lock-only", action="store_true")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-agg", action="store_true")
    p.add_argument("--skip-day-iv", action="store_true")
    p.add_argument("--features", action="store_true", help="继续跑 monthly feature_merge 并切按日特征")
    p.add_argument(
        "--features-root",
        default=str(DEFAULT_FEATURES_ROOT),
        help=(
            "特征/归一化输出根：写入 $ROOT/quote_features_raw 与 "
            "$ROOT/quote_features_test（默认 ~/train_data/dte1_options_old_lock_feat）"
        ),
    )
    p.add_argument(
        "--norm-mode",
        choices=["rolling", "frozen", "none"],
        default="rolling",
        help="rolling=经典离线(window=2000)；frozen=部署/流式同款 npz；none=只写 raw",
    )
    p.add_argument(
        "--frozen-norm",
        default=str(DEFAULT_FROZEN_NORM),
        help="仅 --norm-mode frozen 时使用的 .npz",
    )
    p.add_argument(
        "--feat-history-root",
        default=str(DEFAULT_FEAT_HISTORY),
        help="rolling 预热：从此处借前几个月 raw（同符号/分辨率）",
    )
    p.add_argument("--skip-frozen-norm", action="store_true", help="兼容旧开关；等价 --norm-mode none")
    p.add_argument("--warmup-trading-days", type=int, default=10)
    p.add_argument("--vix-history-months", type=int, default=7)
    p.add_argument(
        "--strict-warmup",
        action="store_true",
        help="预热缺口视为失败（默认只写报告并警告）",
    )
    p.add_argument("--skip-warmup-check", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument(
        "--stock-resamp",
        default=str(DEFAULT_STOCK_RESAMP),
        help="股价 1min 根目录（缺数/预热/day_iv 用）",
    )
    p.add_argument(
        "--feature-config",
        default=str(_REPO_ROOT / "qqq_btc/CONFIG/slow_feature_qqq_v2.json"),
    )
    p.add_argument(
        "--norm-feature-config",
        default=str(_REPO_ROOT / "qqq_btc/CONFIG/slow_feature_qqq_v4.json"),
        help="frozen_norm 时读取 feature name 列表的配置",
    )
    p.add_argument("--db-path", default="/home/kingfang007/notebook/stocks.db")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    py = args.python
    if not Path(py).exists():
        py = sys.executable
    exp = Path(args.exp).expanduser()
    exp.mkdir(parents=True, exist_ok=True)
    map_path = exp / "locked_targets_map_open_4bucket.parquet"
    report_path = exp / "lock_report.json"
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    symbol = symbols[0]

    # 1) lock from open
    lock_cmd = [
        py,
        str(_SCRIPT_DIR / "step1_lock_4bucket_from_open.py"),
        "--config",
        args.config,
        "--symbols",
        ",".join(symbols),
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--output",
        str(map_path),
        "--report",
        str(report_path),
        "--assume-iv",
        str(args.assume_iv),
    ]
    _run(lock_cmd)
    if args.lock_only:
        stage_by_date(
            exp,
            map_path,
            symbol=symbol,
            dates=sorted(pd.read_parquet(map_path)["date_str"].astype(str).unique()),
        )
        logger.info("lock-only done → %s", map_path)
        return 0

    # exp 本身就是 raw_1s 根（与 step2 默认 OUTPUT_DIR 一致）
    raw_1s = exp
    opt_1m = exp / "options_1m"
    day_iv = exp / "quote_options_day_iv"

    # 2) quote download
    if not args.skip_download:
        sniper = [
            py,
            str(_SCRIPT_DIR / "step2_polygon_second_sniper_v1.py"),
            "--target-map",
            str(map_path),
            "--output-dir",
            str(raw_1s),
            "--stock-output-dir",
            str(Path("/mnt/s990/data/raw_1s/stocks")),
            "--start-date",
            args.start_date,
            "--end-date",
            args.end_date,
            "--symbols",
            ",".join(symbols),
            "--max-workers",
            str(args.max_workers),
            "--no-download-stock",
        ]
        if args.force:
            sniper.append("--force")
        _run(sniper)

    # 3) 1s → 1m
    if not args.skip_agg:
        agg = [
            py,
            str(_SCRIPT_DIR / "step3_databento_aggregate_1s_to_1m.py"),
            "--input-dir",
            str(raw_1s),
            "--output-dir",
            str(opt_1m),
            "--symbol",
            symbol,
            "--date-from",
            args.start_date,
            "--date-to",
            args.end_date,
        ]
        if args.force:
            agg.append("--force")
        _run(agg)

    # 4) day_iv
    if not args.skip_day_iv:
        env = {
            "BACKFILL_OPT_1M": str(opt_1m),
            "BACKFILL_DAY_IV": str(day_iv),
            "BACKFILL_STOCK_RESAMP": str(Path(args.stock_resamp).expanduser()),
            "BACKFILL_DB": args.db_path,
        }
        code = r"""
import multiprocessing
from pathlib import Path
import os
from preprocess.ask_bid.option_cac_day_vectorized_day import OptionIVCalculator
try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass
calc = OptionIVCalculator(
    db_path=os.environ["BACKFILL_DB"],
    option_root=os.environ["BACKFILL_OPT_1M"],
    data_root=os.environ["BACKFILL_STOCK_RESAMP"],
    iv_option_root=os.environ["BACKFILL_DAY_IV"],
)
calc.run(max_concurrent_stocks=2)
print("day_iv root", os.environ["BACKFILL_DAY_IV"])
"""
        _run([py, "-c", code], env=env)

    # 5) optional monthly features + frozen_norm
    warmup_report: dict[str, Any] = {}
    if args.features:
        warmup_path = exp / "warmup_report.json"
        if not args.skip_warmup_check:
            warm_cmd = [
                py,
                str(_SCRIPT_DIR / "backfill_warmup_check.py"),
                "--start-date",
                args.start_date,
                "--end-date",
                args.end_date,
                "--symbols",
                ",".join(symbols),
                "--stock-root",
                str(Path(args.stock_resamp).expanduser()),
                "--warmup-trading-days",
                str(args.warmup_trading_days),
                "--vix-history-months",
                str(args.vix_history_months),
                "--report",
                str(warmup_path),
            ]
            if args.strict_warmup:
                warm_cmd.append("--strict")
            try:
                _run(warm_cmd)
            except subprocess.CalledProcessError as exc:
                logger.error("warmup check failed (strict): %s", exc)
                raise
            if warmup_path.is_file():
                warmup_report = json.loads(warmup_path.read_text(encoding="utf-8"))
                if warmup_report.get("blockers"):
                    for b in warmup_report["blockers"]:
                        logger.error("WARMUP BLOCKER: %s", b)
                    if args.strict_warmup:
                        raise SystemExit(
                            "strict-warmup: 过去数据中断/缺数，已中止特征生成。"
                            f" 详见 {warmup_path}"
                        )
                    logger.warning(
                        "warmup 有缺口但仍继续（未开 --strict-warmup）。报告: %s",
                        warmup_path,
                    )
                for w in warmup_report.get("warnings") or []:
                    logger.warning("WARMUP WARN: %s", w)

        env = {
            "BACKFILL_EXP": str(exp),
            "BACKFILL_FEAT_ROOT": str(Path(args.features_root).expanduser()),
            "BACKFILL_FEAT_CFG": str(Path(args.feature_config).expanduser()),
            "BACKFILL_NORM_CFG": str(Path(args.norm_feature_config).expanduser()),
            "BACKFILL_START": args.start_date,
            "BACKFILL_END": args.end_date,
            "BACKFILL_SYM": symbol,
            "BACKFILL_FROZEN": str(Path(args.frozen_norm).expanduser()),
            "BACKFILL_NORM_MODE": (
                "none"
                if args.skip_frozen_norm
                else str(args.norm_mode)
            ),
            "BACKFILL_FEAT_HISTORY": str(Path(args.feat_history_root).expanduser()),
        }
        code = r"""
import json, glob, os, shutil
from pathlib import Path
import pandas as pd
from preprocess.ask_bid.iv_day2month import process_single_symbol
import preprocess.ask_bid.feature_merge_option_raw as fm
from preprocess.ask_bid.options_locked_feature import process_single_file
from qqq_btc.common.frozen_norm import apply_frozen_norm_df
from preprocess.ask_bid.apply_rolling_norm_standalone import (
    load_target_features,
    process_single_directory,
)

exp = Path(os.environ["BACKFILL_EXP"])
feat_root = Path(os.environ.get("BACKFILL_FEAT_ROOT") or exp)
sym = os.environ["BACKFILL_SYM"]
start = os.environ["BACKFILL_START"]
end = os.environ["BACKFILL_END"]
months = [p.strftime("%Y-%m") for p in pd.period_range(start[:7], end[:7], freq="M")]
norm_mode = os.environ.get("BACKFILL_NORM_MODE", "rolling")
hist_root = Path(os.environ.get("BACKFILL_FEAT_HISTORY", ""))
inp = exp / "quote_options_day_iv"
out_m = feat_root / "quote_options_monthly_iv"
files = sorted(glob.glob(f"{inp}/{sym}/**/*.parquet", recursive=True))
print("day_iv files", len(files))
print("features_root", feat_root)
print("feature months", months)
if files:
    print(process_single_symbol((sym, files, str(out_m))))
bucketed = feat_root / "quote_options_bucketed_v7"
fm.OPTION_MONTHLY_DIR = out_m
fm.AGG_OPTION_MONTHLY_DIR = bucketed
fm.OUTPUT_FEATURES_DIR = feat_root / "quote_features_raw"
cfg = json.loads(Path(os.environ["BACKFILL_FEAT_CFG"]).read_text())
for ym in months:
    raw_month = out_m / sym / "standard" / f"{ym}.parquet"
    if raw_month.is_file():
        print(process_single_file((raw_month, bucketed, sym)) or f"bucketed ok {ym}")
    else:
        print("WARN missing monthly_iv", raw_month)
    print(fm.process_stock_month(sym, ym, cfg))

def _prev_ym(ym0: str, k: int = 1) -> str:
    y, m = [int(x) for x in ym0.split("-")]
    m -= k
    while m <= 0:
        m += 12
        y -= 1
    return f"{y:04d}-{m:02d}"

def _stage_test_with_history(res: str) -> Path | None:
    # 把 quote_features_raw 的目标月拷到 quote_features_test，并借前 2 月做 rolling buffer。
    # 注意: process_single_directory 会原地改写 test 目录下所有 parquet（含借来的历史月）。
    # raw 目录保持未归一化；test = 归一化后（历史月仅作窗口，不是本轮新特征）。
    raw_leaf = feat_root / f"quote_features_raw/{sym}/regular/09:30-16:00/{res}"
    test_leaf = feat_root / f"quote_features_test/{sym}/regular/09:30-16:00/{res}"
    test_leaf.mkdir(parents=True, exist_ok=True)
    copied = 0
    for ym in months:
        raw_p = raw_leaf / f"{ym}.parquet"
        if not raw_p.is_file():
            print("skip norm missing raw", raw_p)
            continue
        shutil.copy2(raw_p, test_leaf / f"{ym}.parquet")
        copied += 1
        print("stage raw->test", raw_p.name)
    if copied == 0:
        return None
    # rolling buffer: 区间首月之前再借 2 个月（优先本 features_root raw，再 hist）
    first = months[0]
    for k in (1, 2):
        pym = _prev_ym(first, k)
        src_candidates = [
            raw_leaf / f"{pym}.parquet",
            hist_root / f"{sym}/regular/09:30-16:00/{res}/{pym}.parquet",
        ]
        for src in src_candidates:
            if src.is_file():
                shutil.copy2(src, test_leaf / f"{pym}.parquet")
                print("rolling history seed", src, "->", test_leaf / f"{pym}.parquet")
                break
    return test_leaf

if norm_mode == "frozen":
    frozen = Path(os.environ["BACKFILL_FROZEN"])
    norm_cfg = json.loads(Path(os.environ["BACKFILL_NORM_CFG"]).read_text())
    names = None
    feats = norm_cfg.get("features") or norm_cfg.get("slow_features") or []
    if isinstance(feats, list) and feats and isinstance(feats[0], dict):
        names = [f.get("name") for f in feats if f.get("name")]
    elif isinstance(feats, list):
        names = [str(x) for x in feats]
    for res in ("1min", "5min"):
        for ym in months:
            raw_p = feat_root / f"quote_features_raw/{sym}/regular/09:30-16:00/{res}/{ym}.parquet"
            if not raw_p.is_file():
                print("skip frozen missing", raw_p)
                continue
            out_dir = feat_root / f"quote_features_test/{sym}/regular/09:30-16:00/{res}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_p = out_dir / f"{ym}.parquet"
            df = pd.read_parquet(raw_p)
            normed = apply_frozen_norm_df(df, frozen, feature_names=names)
            normed.to_parquet(out_p, index=False)
            print("frozen_norm wrote", out_p, "rows", len(normed))
elif norm_mode == "rolling":
    # 与 DATA_PIPELINE / apply_rolling_norm_standalone 同口径：window=2000, min_periods=100
    os.environ["FEATURE_CONFIG"] = os.environ["BACKFILL_FEAT_CFG"]
    norm_cols = load_target_features(Path(os.environ["BACKFILL_FEAT_CFG"]))
    print("rolling_norm cols", len(norm_cols), "config", os.environ["BACKFILL_FEAT_CFG"])
    for res in ("1min", "5min"):
        leaf = _stage_test_with_history(res)
        if leaf is None:
            continue
        msg = process_single_directory((leaf, norm_cols))
        print("rolling_norm", res, msg)
        if msg and str(msg).startswith("ERROR"):
            raise RuntimeError(msg)
else:
    print("norm_mode=none: keep quote_features_raw only under", feat_root)
"""
        _run([py, "-c", code], env=env)

    start, end = args.start_date, args.end_date
    dates = sorted(
        pd.read_parquet(map_path)
        .query("date_str >= @start and date_str <= @end")["date_str"]
        .astype(str)
        .unique()
    )
    feat_root = Path(args.features_root).expanduser()
    staged = stage_by_date(
        exp, map_path, symbol=symbol, dates=dates, features_root=feat_root
    )
    norm_mode_eff = "none" if args.skip_frozen_norm else args.norm_mode
    summary = {
        "exp": str(exp),
        "features_root": str(feat_root),
        "quote_features_raw": str(feat_root / "quote_features_raw"),
        "quote_features_test": str(feat_root / "quote_features_test"),
        "map": str(map_path),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "symbols": symbols,
        "lock_mode": "open_price_4bucket",
        "features": bool(args.features),
        "norm_mode": norm_mode_eff if args.features else None,
        "frozen_norm": (
            str(Path(args.frozen_norm).expanduser())
            if (args.features and norm_mode_eff == "frozen")
            else None
        ),
        "feat_history_root": (
            str(Path(args.feat_history_root).expanduser())
            if (args.features and norm_mode_eff == "rolling")
            else None
        ),
        "warmup_report": str(exp / "warmup_report.json") if (exp / "warmup_report.json").is_file() else None,
        "warmup_ok": warmup_report.get("ok") if warmup_report else None,
        "warmup_blockers": warmup_report.get("blockers") if warmup_report else [],
        "days": staged,
        "note": (
            "rolling=经典离线 apply_rolling_norm_standalone(window=2000)；"
            "frozen=流式/deploy 同款 frozen_norm_qqq_daily.npz"
        ),
    }
    (exp / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("DONE exp=%s by_date days=%d", exp, len(staged))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
