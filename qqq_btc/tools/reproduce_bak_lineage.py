#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
复现 bak（_bak_pre4c）期权特征血缘。

根因：bak monthly 使用 locked_targets_map_0dte_dynamic 的
primary + secondary（prefer_primary_gapfill）；现网只用 v3 单合约。

步骤:
  1) 从 v3(primary) + old/fixed8(secondary) 组装 dual 1m
  2) prefer_primary_gapfill
  3) day_iv → monthly → bucketed
  4) 与 bak monthly / bucketed 对账

用法:
  python qqq_btc/tools/reproduce_bak_lineage.py assemble \\
      --date-from 2025-07-01 --date-to 2025-12-31
  python qqq_btc/tools/reproduce_bak_lineage.py day-iv \\
      --date-from 2025-07-01 --date-to 2025-12-31
  python qqq_btc/tools/reproduce_bak_lineage.py monthly-bucketed \\
      --months 2025-07,2025-08,2025-09,2025-10,2025-11,2025-12
  python qqq_btc/tools/reproduce_bak_lineage.py validate-monthly --month 2025-08
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qqq_btc.tools.bak_monthly_switch_logic import (  # noqa: E402
    load_role_map,
    primary_lookup,
)

logger = logging.getLogger("reproduce_bak_lineage")

NY = "America/New_York"
DYNAMIC_MAP = Path.home() / "train_data/locked_targets_map_0dte_dynamic.parquet"
OUT_ROOT = Path.home() / "train_data/bak_lineage_reproduce"
SRC_PRIMARY = Path("/mnt/s990/data/raw_1m/options_databento_v3")
SRC_OLD = Path("/mnt/s990/data/raw_1m/options_databento_old")
SRC_FIXED8 = Path("/mnt/s990/data/raw_1m/options_databento_fixed8_corrected")
SRC_CANON = Path("/mnt/s990/data/raw_1m/options_databento")
BAK_MONTHLY = Path.home() / "train_data/_bak_pre4c/quote_options_monthly_iv_QQQ/standard"
BAK_BUCKETED = Path.home() / "train_data/_bak_pre4c/quote_options_bucketed_v7_QQQ"


def _norm_ts(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    if col not in out.columns and "ts" in out.columns:
        out[col] = out["ts"]
    out[col] = pd.to_datetime(out[col])
    if out[col].dt.tz is None:
        out[col] = out[col].dt.tz_localize(NY)
    else:
        out[col] = out[col].dt.tz_convert(NY)
    return out


def _norm_ticker(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("O:", "", regex=False)


def _load_day_quotes(root: Path, symbol: str, day: str) -> Optional[pd.DataFrame]:
    fp = root / symbol / f"{symbol}_{day}.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    if df.empty:
        return None
    df = _norm_ts(df)
    df["ticker"] = _norm_ticker(df["ticker"])
    df["bucket_id"] = df["bucket_id"].astype(int)
    return df


def assemble_day_dual(
    day: str,
    role_day: pd.DataFrame,
    symbol: str = "QQQ",
) -> tuple[Optional[pd.DataFrame], dict]:
    """按 dynamic map 的 (ticker, bucket) 从多源拼接 dual 1m，再 prefer-primary gapfill。

    注意：同一合约可同时出现在多个 bucket（盘中换约），绝不能按 ticker 单独覆写
    bucket_id，否则会把 primary 行挪到别的 bucket，只剩 secondary。
    """
    role_day = role_day.copy()
    role_day["contract_symbol"] = (
        role_day["contract_symbol"].astype(str).str.replace("O:", "", regex=False)
    )
    wanted_pairs = {
        (str(r.contract_symbol), int(r.bucket_id)) for r in role_day.itertuples()
    }
    wanted = {t for t, _ in wanted_pairs}
    if "date_str" not in role_day.columns:
        role_day = role_day.assign(date_str=day)
    prim = primary_lookup(role_day)

    parts = []
    sources_hit = []
    for root, tag in [
        (SRC_PRIMARY, "v3"),
        (SRC_CANON, "canon"),
        (SRC_OLD, "old"),
        (SRC_FIXED8, "fixed8"),
    ]:
        df = _load_day_quotes(root, symbol, day)
        if df is None:
            continue
        # 只保留 map 声明的 (ticker, bucket) 对，保留源文件 bucket 标签
        pair_ok = [
            (str(t), int(b)) in wanted_pairs
            for t, b in zip(df["ticker"], df["bucket_id"])
        ]
        sub = df.loc[pair_ok]
        if len(sub) == 0:
            continue
        sub = sub.copy()
        sub["_src"] = tag
        parts.append(sub)
        sources_hit.append(tag)

    if not parts:
        return None, {
            "day": day,
            "status": "missing_all_sources",
            "wanted": sorted(wanted),
            "wanted_pairs": sorted(f"{t}|{b}" for t, b in wanted_pairs),
        }

    raw = pd.concat(parts, ignore_index=True)
    # 同一 (ts,bucket,ticker) 多源重复时优先 v3/canon
    src_rank = {"v3": 0, "canon": 1, "old": 2, "fixed8": 3}
    raw["_sr"] = raw["_src"].map(src_rank).fillna(9)
    raw = raw.sort_values(["timestamp", "bucket_id", "ticker", "_sr"])
    raw = raw.drop_duplicates(["timestamp", "bucket_id", "ticker"], keep="first")

    # prefer-primary: primary 全部分钟 ∪ secondary 仅 gap 分钟
    rebuilt_parts = []
    for b in sorted(int(x) for x in raw["bucket_id"].unique()):
        p_sym = prim.get((day, b))
        sub = raw[raw["bucket_id"] == b]
        if p_sym is None:
            rebuilt_parts.append(sub)
            continue
        prim_rows = sub[sub["ticker"] == p_sym]
        prim_ts = set(prim_rows["timestamp"])
        sec_rows = sub[sub["ticker"] != p_sym]
        sec_gap = sec_rows[~sec_rows["timestamp"].isin(prim_ts)]
        rebuilt_parts.append(pd.concat([prim_rows, sec_gap], ignore_index=True))

    out = pd.concat(rebuilt_parts, ignore_index=True)
    out = out.drop(columns=[c for c in ("_src", "_sr") if c in out.columns])
    out = out.sort_values(["timestamp", "bucket_id", "ticker"]).reset_index(drop=True)

    found_pairs = {(str(t), int(b)) for t, b in zip(out["ticker"], out["bucket_id"])}
    is_prim = pd.Series(
        [
            str(t) == prim.get((day, int(b)))
            for t, b in zip(out["ticker"], out["bucket_id"])
        ],
        index=out.index,
    )
    meta = {
        "day": day,
        "status": "ok",
        "sources": sources_hit,
        "wanted": sorted(wanted),
        "found": sorted({t for t, _ in found_pairs}),
        "missing_pairs": sorted(f"{t}|{b}" for t, b in (wanted_pairs - found_pairs)),
        "missing_tickers": sorted(wanted - {t for t, _ in found_pairs}),
        "rows": int(len(out)),
        "n_tickers": int(out["ticker"].nunique()),
        "n_primary_rows": int(is_prim.sum()),
        "n_secondary_rows": int((~is_prim).sum()),
    }
    return out, meta


def cmd_assemble(args: argparse.Namespace) -> None:
    out_1m = Path(args.out_root) / "raw_1m_prefer_primary"
    out_1m.mkdir(parents=True, exist_ok=True)
    (out_1m / args.symbol).mkdir(parents=True, exist_ok=True)

    role = load_role_map(Path(args.dynamic_map), args.date_from, args.date_to)
    days = sorted(role["date_str"].unique())
    stats = {"days_ok": 0, "days_fail": 0, "rows": 0, "missing_ticker_days": [], "per_day": []}

    for day in days:
        role_day = role[role["date_str"] == day].copy()
        df, meta = assemble_day_dual(day, role_day, symbol=args.symbol)
        stats["per_day"].append(meta)
        if df is None:
            stats["days_fail"] += 1
            logger.warning("assemble fail %s: %s", day, meta)
            continue
        miss = meta.get("missing_pairs") or meta.get("missing_tickers") or []
        if miss:
            stats["missing_ticker_days"].append({"day": day, "missing": miss})
        fp = out_1m / args.symbol / f"{args.symbol}_{day}.parquet"
        df.to_parquet(fp, index=False)
        stats["days_ok"] += 1
        stats["rows"] += meta["rows"]
        logger.info(
            "%s rows=%d tickers=%d sec_rows=%d missing=%s src=%s",
            day, meta["rows"], meta["n_tickers"], meta["n_secondary_rows"],
            miss, meta["sources"],
        )

    summary_path = Path(args.out_root) / "assemble_summary.json"
    # shrink per_day for summary file
    slim = {k: v for k, v in stats.items() if k != "per_day"}
    slim["n_days"] = len(days)
    slim["n_missing_ticker_days"] = len(stats["missing_ticker_days"])
    slim["missing_ticker_days"] = stats["missing_ticker_days"][:50]
    summary_path.write_text(json.dumps(slim, indent=2, ensure_ascii=False))
    (Path(args.out_root) / "assemble_per_day.json").write_text(
        json.dumps(stats["per_day"], indent=2, ensure_ascii=False)
    )
    print(json.dumps(slim, indent=2, ensure_ascii=False))


def cmd_day_iv(args: argparse.Namespace) -> None:
    """对组装好的 1m 跑 day_iv（只处理区间内日期）。"""
    import multiprocessing
    from preprocess.ask_bid.option_cac_day_vectorized_day import OptionIVCalculator

    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass

    opt_root = Path(args.out_root) / "raw_1m_prefer_primary"
    iv_root = Path(args.out_root) / "quote_options_day_iv"
    iv_root.mkdir(parents=True, exist_ok=True)

    # OptionIVCalculator 默认扫整个 option_root；这里临时只保留区间文件的工作副本已在 out 中
    calc = OptionIVCalculator(
        db_path=str(Path.home() / "notebook/stocks.db"),
        option_root=str(opt_root),
        data_root=str(Path.home() / "train_data/spnq_train_resampled"),
        iv_option_root=str(iv_root),
    )
    # 若已有文件则跳过；先删区间内旧文件以便重算
    sym_iv = iv_root / args.symbol / "standard"
    if args.force and sym_iv.exists():
        role = load_role_map(Path(args.dynamic_map), args.date_from, args.date_to)
        for day in role["date_str"].unique():
            fp = sym_iv / f"{args.symbol}_{day}.parquet"
            if fp.exists():
                fp.unlink()

    calc.run(max_concurrent_stocks=1)
    n = len(list(sym_iv.glob("*.parquet"))) if sym_iv.exists() else 0
    print(json.dumps({"day_iv_files": n, "iv_root": str(iv_root)}, indent=2))


def cmd_monthly_bucketed(args: argparse.Namespace) -> None:
    """day_iv → monthly → bucketed（仅指定月份）。"""
    from preprocess.ask_bid.iv_day2month import process_single_symbol
    from preprocess.ask_bid.options_locked_feature import process_single_file

    months = [m.strip() for m in args.months.split(",") if m.strip()]
    iv_root = Path(args.out_root) / "quote_options_day_iv" / args.symbol / "standard"
    monthly_root = Path(args.out_root) / "quote_options_monthly_iv" / args.symbol / "standard"
    bucketed_root = Path(args.out_root) / "quote_options_bucketed_v7"
    monthly_root.mkdir(parents=True, exist_ok=True)
    bucketed_root.mkdir(parents=True, exist_ok=True)

    # gather day files by month
    day_files = sorted(iv_root.glob(f"{args.symbol}_*.parquet"))
    by_month: dict[str, list] = {}
    for fp in day_files:
        day = fp.stem.split("_")[-1]
        mo = day[:7]
        if mo in months:
            by_month.setdefault(mo, []).append(str(fp))

    for mo in months:
        files = by_month.get(mo, [])
        if not files:
            logger.warning("no day_iv for month %s", mo)
            continue
        # process_single_symbol expects (symbol, files, output_base)
        # output_base is quote_options_monthly_iv parent of symbol/standard
        out_base = str(Path(args.out_root) / "quote_options_monthly_iv")
        msg = process_single_symbol((args.symbol, files, out_base))
        logger.info("monthly %s: %s files=%d", mo, msg, len(files))

    for mo in months:
        src = monthly_root / f"{mo}.parquet"
        if not src.exists():
            logger.warning("missing monthly %s", src)
            continue
        msg = process_single_file((src, bucketed_root, args.symbol))
        logger.info("bucketed %s: %s", mo, msg)

    print(json.dumps({"months": months, "monthly_root": str(monthly_root), "bucketed_root": str(bucketed_root)}, indent=2))


def cmd_feature_merge(args: argparse.Namespace) -> None:
    """用复现的 monthly/bucketed 跑 H2 feature merge（1min+5min）。"""
    import preprocess.ask_bid.feature_merge_option_raw as fm

    months = [m.strip() for m in args.months.split(",") if m.strip()]
    out_root = Path(args.out_root)
    cfg_path = Path(args.feature_config)
    with open(cfg_path, encoding="utf-8") as f:
        config = json.load(f)

    fm.OPTION_MONTHLY_DIR = out_root / "quote_options_monthly_iv"
    fm.AGG_OPTION_MONTHLY_DIR = out_root / "quote_options_bucketed_v7"
    fm.OUTPUT_FEATURES_DIR = out_root / "quote_features_raw"
    fm.CONFIG_FILE = str(cfg_path)
    fm.OVERWRITE_EXISTING = True
    fm.OUTPUT_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for mo in months:
        msg = fm.process_stock_month(args.symbol, mo, config)
        results.append({"month": mo, "msg": str(msg)})
        logger.info("feature_merge %s: %s", mo, msg)

    # 补 VIX / cat / labels（与 feature_merge __main__ 后半段对齐）
    try:
        fm.generate_vix_level_global(config)
    except Exception as e:
        logger.warning("generate_vix_level_global: %s", e)
    try:
        fm.update_vol_vix_abs(config)
    except Exception as e:
        logger.warning("update_vol_vix_abs: %s", e)
    try:
        fm.update_cat_features_in_files(config)
    except Exception as e:
        logger.warning("update_cat_features_in_files: %s", e)
    try:
        fm.update_new_labels_in_files(config)
    except Exception as e:
        logger.warning("update_new_labels_in_files: %s", e)

    report = {
        "months": months,
        "results": results,
        "raw_root": str(fm.OUTPUT_FEATURES_DIR),
        "config": str(cfg_path),
    }
    (out_root / "feature_merge_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cmd_rolling_norm(args: argparse.Namespace) -> None:
    """bak_raw 预 H2 + 复现 H2 raw → rolling norm → quote_features_train。"""
    import shutil
    import preprocess.ask_bid.apply_rolling_norm_standalone as norm

    out_root = Path(args.out_root)
    months_h2 = [m.strip() for m in args.months.split(",") if m.strip()]
    bak_raw_1m = Path.home() / "train_data/_bak_pre4c/quote_features_raw_QQQ/regular/09:30-16:00/1min"
    bak_raw_5m = Path.home() / "train_data/_bak_pre4c/quote_features_raw_QQQ/regular/09:30-16:00/5min"
    rep_raw = out_root / "quote_features_raw" / args.symbol / "regular" / "09:30-16:00"
    train_root = out_root / "quote_features_train" / args.symbol / "regular" / "09:30-16:00"

    for res, bak_src in [("1min", bak_raw_1m), ("5min", bak_raw_5m)]:
        dst = train_root / res
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        # pre-H2 from bak raw
        if bak_src.exists():
            for p in sorted(bak_src.glob("*.parquet")):
                if p.stem < months_h2[0]:
                    shutil.copy2(p, dst / p.name)
        # H2 from reproduced raw
        src = rep_raw / res
        for mo in months_h2:
            sp = src / f"{mo}.parquet"
            if not sp.exists():
                logger.warning("missing reproduced raw %s", sp)
                continue
            shutil.copy2(sp, dst / f"{mo}.parquet")

    import os

    os_environ_cfg = str(Path(args.feature_config).resolve())
    os.environ["FEATURE_CONFIG"] = os_environ_cfg
    norm.CONFIG_PATH = Path(os_environ_cfg)
    targets = norm.load_target_features(norm.CONFIG_PATH)
    msgs = []
    for res in ("1min", "5min"):
        d = train_root / res
        msg = norm.process_single_directory((d, targets))
        msgs.append({"res": res, "msg": str(msg), "n_files": len(list(d.glob("*.parquet")))})
        logger.info("rolling_norm %s: %s", res, msg)

    # compare H2 vs bak_train options
    bak_train = Path.home() / "train_data/_bak_pre4c/quote_features_train_QQQ/regular/09:30-16:00/1min"
    cmp = {}
    for mo in months_h2:
        a = pd.read_parquet(bak_train / f"{mo}.parquet")
        b = pd.read_parquet(train_root / "1min" / f"{mo}.parquet")
        a = _norm_ts(a).sort_values("timestamp").reset_index(drop=True)
        b = _norm_ts(b).sort_values("timestamp").reset_index(drop=True)
        n = min(len(a), len(b))
        a, b = a.iloc[:n], b.iloc[:n]
        opts = [c for c in a.columns if c.startswith("options_") and c in b.columns]
        corrs = []
        for c in opts:
            x = pd.to_numeric(a[c], errors="coerce")
            y = pd.to_numeric(b[c], errors="coerce")
            m = x.notna() & y.notna()
            if m.sum() < 50 or x[m].std() == 0 or y[m].std() == 0:
                continue
            corrs.append(float(np.corrcoef(x[m], y[m])[0, 1]))
        cmp[mo] = {
            "n": n,
            "n_opt": len(corrs),
            "med_corr": float(np.median(corrs)) if corrs else None,
            "min_corr": float(np.min(corrs)) if corrs else None,
        }

    report = {"msgs": msgs, "vs_bak_train": cmp, "train_root": str(train_root)}
    (out_root / "rolling_norm_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cmd_split_norm(args: argparse.Namespace) -> None:
    """raw → split train/val/test → 各 stage 独立 rolling norm。"""
    import os
    import shutil
    import preprocess.ask_bid.apply_rolling_norm_standalone as norm

    out_root = Path(args.out_root)
    symbol = args.symbol
    raw_root = out_root / "quote_features_raw" / symbol
    if not raw_root.exists():
        raise SystemExit(f"missing raw features: {raw_root}")

    train_range = (pd.Timestamp("2023-03-01"), pd.Timestamp("2025-12-31"))
    val_range = (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31"))
    test_range = (pd.Timestamp("2026-04-01"), pd.Timestamp("2026-06-30"))

    stage_roots = {
        "train": out_root / "quote_features_train" / symbol,
        "val": out_root / "quote_features_val" / symbol,
        "test": out_root / "quote_features_test" / symbol,
    }
    for d in stage_roots.values():
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    copied = {"train": 0, "val": 0, "test": 0, "skip": 0}
    for fp in sorted(raw_root.rglob("*.parquet")):
        try:
            ym = pd.Timestamp(fp.stem + "-01")
        except Exception:
            copied["skip"] += 1
            continue
        if train_range[0] <= ym <= train_range[1]:
            stage = "train"
        elif val_range[0] <= ym <= val_range[1]:
            stage = "val"
        elif test_range[0] <= ym <= test_range[1]:
            stage = "test"
        else:
            copied["skip"] += 1
            continue
        rel = fp.relative_to(raw_root)
        dst = stage_roots[stage] / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fp, dst)
        copied[stage] += 1

    os.environ["FEATURE_CONFIG"] = str(Path(args.feature_config).resolve())
    norm.CONFIG_PATH = Path(os.environ["FEATURE_CONFIG"])
    targets = norm.load_target_features(norm.CONFIG_PATH)
    msgs = []
    for stage, root in stage_roots.items():
        for leaf in norm.find_leaf_directories(root):
            msg = norm.process_single_directory((leaf, targets))
            n = len(list(leaf.glob("*.parquet")))
            msgs.append({"stage": stage, "dir": str(leaf), "msg": str(msg), "n_files": n})
            logger.info("norm %s %s: %s", stage, leaf, msg)

    report = {
        "copied": copied,
        "norm": msgs,
        "train_root": str(stage_roots["train"]),
        "val_root": str(stage_roots["val"]),
        "test_root": str(stage_roots["test"]),
    }
    (out_root / "split_norm_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cmd_label_stages(args: argparse.Namespace) -> None:
    """对 train/val/test 1min 分别跑 label_pipeline。"""
    import subprocess

    out_root = Path(args.out_root)
    reports = {}
    for stage in ("train", "val", "test"):
        feat_dir = (
            out_root
            / f"quote_features_{stage}"
            / args.symbol
            / "regular"
            / "09:30-16:00"
            / "1min"
        )
        if not feat_dir.exists() or not list(feat_dir.glob("*.parquet")):
            logger.warning("skip label %s: empty %s", stage, feat_dir)
            continue
        report = out_root / f"label_report_{stage}.json"
        cmd = [
            sys.executable,
            str(_REPO / "qqq_btc/tools/label_pipeline.py"),
            "--input",
            str(feat_dir),
            "--output",
            str(feat_dir),
            "--symbol",
            args.symbol,
            "--anchor-config",
            str(Path(args.anchor_config)),
            "--report",
            str(report),
        ]
        logger.info("label %s", stage)
        subprocess.check_call(cmd, cwd=str(_REPO))
        if report.exists():
            reports[stage] = json.loads(report.read_text(encoding="utf-8"))
    print(json.dumps({"stages": list(reports), "avg_net_std": {
        s: reports[s].get("avg_net_std") for s in reports
    }}, indent=2, ensure_ascii=False))


def cmd_label_h2(args: argparse.Namespace) -> None:
    """对复现 train 特征跑 label_pipeline（H2 月）。"""
    import subprocess

    out_root = Path(args.out_root)
    feat_dir = out_root / "quote_features_train" / args.symbol / "regular" / "09:30-16:00" / "1min"
    # label_pipeline 会处理目录下全部月份；预 H2 也一起标无妨
    report = out_root / "label_report_h2.json"
    cmd = [
        sys.executable,
        str(_REPO / "qqq_btc/tools/label_pipeline.py"),
        "--input",
        str(feat_dir),
        "--output",
        str(feat_dir),
        "--symbol",
        args.symbol,
        "--anchor-config",
        str(Path(args.anchor_config)),
        "--report",
        str(report),
    ]
    logger.info("run %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(_REPO))
    print(report.read_text(encoding="utf-8") if report.exists() else "{}")


def cmd_replay_h2(args: argparse.Namespace) -> None:
    """V4 infer + strict replay on reproduced H2 features."""
    import shutil
    import subprocess

    out_root = Path(args.out_root)
    months = [m.strip() for m in args.months.split(",") if m.strip()]
    feat_stage = out_root / "quote_features_train" / args.symbol / "regular" / "09:30-16:00"
    feat_eval = out_root / "h2_eval_features"
    if feat_eval.exists():
        shutil.rmtree(feat_eval)
    for res in ("1min", "5min"):
        dst = feat_eval / args.symbol / "regular" / "09:30-16:00" / res
        dst.mkdir(parents=True, exist_ok=True)
        for mo in months:
            sp = feat_stage / res / f"{mo}.parquet"
            if sp.exists():
                shutil.copy2(sp, dst / f"{mo}.parquet")

    replay_out = Path(args.replay_out)
    replay_out.mkdir(parents=True, exist_ok=True)
    opt_1m = args.option_1m_root or str(out_root / "raw_1m_prefer_primary")
    cmd = [
        sys.executable,
        str(_REPO / "qqq_btc/tools/eval_test_set.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--config",
        str(args.model_config),
        "--feature-root",
        str(feat_eval),
        "--option-1m-root",
        str(opt_1m),
        "--output-dir",
        str(replay_out),
        "--device",
        args.device,
    ]
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    logger.info("run %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(_REPO), env=env)
    summary = json.loads((replay_out / "replay_summary.json").read_text(encoding="utf-8"))
    print(json.dumps({
        "replay_out": str(replay_out),
        "ic": summary.get("label_metrics", {}).get("ic"),
        "acct25": summary.get("total_net_return"),
        "trades": summary.get("trades"),
        "equity_mult": 1.0 + float(summary.get("total_net_return", 0.0)),
    }, indent=2, ensure_ascii=False))


def cmd_validate_monthly(args: argparse.Namespace) -> None:
    mo = args.month
    bak = pd.read_parquet(BAK_MONTHLY / f"{mo}.parquet")
    rep = pd.read_parquet(
        Path(args.out_root) / "quote_options_monthly_iv" / args.symbol / "standard" / f"{mo}.parquet"
    )
    keys = ["timestamp", "bucket_id", "ticker"]
    bak = _norm_ts(bak)
    rep = _norm_ts(rep)
    bak["ticker"] = _norm_ticker(bak["ticker"])
    rep["ticker"] = _norm_ticker(rep["ticker"])

    bak_k = bak.drop_duplicates(keys)
    rep_k = rep.drop_duplicates(keys)
    both = bak_k.merge(rep_k[keys], on=keys, how="inner")
    only_b = bak_k.merge(rep_k[keys], on=keys, how="left", indicator=True)
    only_r = rep_k.merge(bak_k[keys], on=keys, how="left", indicator=True)

    m = bak.merge(rep, on=keys, suffixes=("_b", "_r"))
    eqs = {}
    for c in ["iv", "delta", "bid", "ask", "volume", "strike_price"]:
        cb, cr = f"{c}_b", f"{c}_r"
        if cb in m.columns and cr in m.columns:
            x = pd.to_numeric(m[cb], errors="coerce")
            y = pd.to_numeric(m[cr], errors="coerce")
            mm = x.notna() & y.notna()
            eqs[c] = {
                "eq": float(np.isclose(x[mm], y[mm], atol=1e-6, equal_nan=True).mean()) if mm.sum() else None,
                "corr": float(np.corrcoef(x[mm], y[mm])[0, 1]) if mm.sum() > 50 and x[mm].std() > 0 else None,
            }

    # bucketed
    bak_b = pd.read_parquet(BAK_BUCKETED / f"{mo}.parquet")
    rep_b_path = Path(args.out_root) / "quote_options_bucketed_v7" / args.symbol / f"{mo}.parquet"
    bkt = None
    if rep_b_path.exists():
        rep_b = pd.read_parquet(rep_b_path)
        bak_b = _norm_ts(bak_b)
        rep_b = _norm_ts(rep_b)
        mb = bak_b.merge(rep_b, on="timestamp", suffixes=("_b", "_r"))
        bkt = {}
        for c in [c for c in bak_b.columns if c.startswith("options_")]:
            x = pd.to_numeric(mb[f"{c}_b"], errors="coerce")
            y = pd.to_numeric(mb[f"{c}_r"], errors="coerce")
            mm = x.notna() & y.notna()
            if mm.sum() < 50:
                continue
            bkt[c] = {
                "corr": float(np.corrcoef(x[mm], y[mm])[0, 1]) if x[mm].std() > 0 and y[mm].std() > 0 else None,
                "eq": float(np.isclose(x[mm], y[mm], atol=1e-6).mean()),
            }

    report = {
        "month": mo,
        "bak_rows": int(len(bak)),
        "rep_rows": int(len(rep)),
        "key_inner": int(len(both)),
        "only_bak": int((only_b["_merge"] == "left_only").sum()),
        "only_rep": int((only_r["_merge"] == "left_only").sum()),
        "coverage_bak": float(len(both) / max(len(bak_k), 1)),
        "overlap_equals": eqs,
        "bucketed": bkt,
    }
    out = Path(args.out_root) / f"validate_monthly_{mo}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Reproduce bak option feature lineage")
    p.add_argument("--out-root", default=str(OUT_ROOT))
    p.add_argument("--dynamic-map", default=str(DYNAMIC_MAP))
    p.add_argument("--symbol", default="QQQ")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("assemble")
    a.add_argument("--date-from", default="2025-07-01")
    a.add_argument("--date-to", default="2025-12-31")
    a.set_defaults(func=cmd_assemble)

    d = sub.add_parser("day-iv")
    d.add_argument("--date-from", default="2025-07-01")
    d.add_argument("--date-to", default="2025-12-31")
    d.add_argument("--force", action="store_true")
    d.set_defaults(func=cmd_day_iv)

    m = sub.add_parser("monthly-bucketed")
    m.add_argument("--months", default="2025-07,2025-08,2025-09,2025-10,2025-11,2025-12")
    m.set_defaults(func=cmd_monthly_bucketed)

    v = sub.add_parser("validate-monthly")
    v.add_argument("--month", default="2025-08")
    v.set_defaults(func=cmd_validate_monthly)

    fm = sub.add_parser("feature-merge")
    fm.add_argument("--months", default="2025-07,2025-08,2025-09,2025-10,2025-11,2025-12")
    fm.add_argument(
        "--feature-config",
        default=str(_REPO / "qqq_btc/CONFIG/slow_feature_qqq_v2.json"),
    )
    fm.set_defaults(func=cmd_feature_merge)

    rn = sub.add_parser("rolling-norm")
    rn.add_argument("--months", default="2025-07,2025-08,2025-09,2025-10,2025-11,2025-12")
    rn.add_argument(
        "--feature-config",
        default=str(_REPO / "qqq_btc/CONFIG/slow_feature_qqq_v4.json"),
    )
    rn.set_defaults(func=cmd_rolling_norm)

    sn = sub.add_parser("split-norm")
    sn.add_argument(
        "--feature-config",
        default=str(_REPO / "qqq_btc/CONFIG/slow_feature_qqq_v4.json"),
    )
    sn.set_defaults(func=cmd_split_norm)

    lb = sub.add_parser("label-h2")
    lb.add_argument(
        "--anchor-config",
        default=str(_REPO / "qqq_btc/CONFIG/anchor_qqq_0dte.json"),
    )
    lb.set_defaults(func=cmd_label_h2)

    ls = sub.add_parser("label-stages")
    ls.add_argument(
        "--anchor-config",
        default=str(_REPO / "qqq_btc/CONFIG/anchor_qqq_0dte.json"),
    )
    ls.set_defaults(func=cmd_label_stages)

    rp = sub.add_parser("replay-h2")
    rp.add_argument("--months", default="2025-07,2025-08,2025-09,2025-10,2025-11,2025-12")
    rp.add_argument(
        "--checkpoint",
        default=str(_REPO / "checkpoint/checkpoints_qqq_v4/best.pth"),
    )
    rp.add_argument(
        "--model-config",
        default=str(_REPO / "qqq_btc/CONFIG/slow_feature_qqq_v4.json"),
    )
    rp.add_argument(
        "--replay-out",
        default=str(_REPO / "qqq_btc/results/v4_replay_2025h2_bak_reproduce"),
    )
    rp.add_argument("--option-1m-root", default=None)
    rp.add_argument("--device", default="cuda")
    rp.set_defaults(func=cmd_replay_h2)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
