#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Honest Gate1/2 快路径 —— 进程内 1s → FCS，无 Redis pitcher/SE/OMS 全栈。

借鉴 s2 turbo：逐秒 process_market_data，分钟边界才 run_compute_cycle；
从 cached_batch_raw 抽 pre-norm 向量做 Gate1，可选 debug_slow 做 Gate2。

用法:
  python qqq_btc/tools/honest_gate1_fcs_fast.py \\
    --dates 2026-07-01,2026-07-02,2026-07-06,2026-07-07,2026-07-08,2026-07-09,2026-07-10 \\
    --option-root /mnt/s990/data/v4_original_jul5/databento_july_w1_openwin/raw_1s \\
    --greek-root ~/train_data/july_w1_v4_honest_openwin/quote_options_day_iv \\
    --offline-raw ~/train_data/july_w1_v4_honest_openwin/quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-07.parquet \\
    --out qqq_btc/results/july_w1_ft56_honest_gate1_fast/
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

_REPO = Path(__file__).resolve().parents[2]
_BASELINE = _REPO / "New_Pro" / "baseline_qqq"
_DAO = _BASELINE / "DAO"

NY_TZ = pytz.timezone("America/New_York")
DEFAULT_DATES = (
    "2026-07-01,2026-07-02,2026-07-06,2026-07-07,"
    "2026-07-08,2026-07-09,2026-07-10"
)
DEFAULT_OPT_ROOT = Path("/mnt/s990/data/v4_original_jul5/databento_july_w1_openwin/raw_1s")
DEFAULT_HONEST = Path.home() / "train_data/july_w1_v4_honest_openwin"
DEFAULT_GREEK = DEFAULT_HONEST / "quote_options_day_iv"
DEFAULT_OFFLINE_RAW = (
    DEFAULT_HONEST / "quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-07.parquet"
)
DEFAULT_OFFLINE_NORM = (
    DEFAULT_HONEST / "quote_features_test/QQQ/regular/09:30-16:00/1min/2026-07.parquet"
)
DEFAULT_STOCK_ROOT = Path("/mnt/s990/data/raw_1s/stocks")
DEFAULT_STOCK_FALLBACKS = {
    "QQQ": Path.home() / "train_data/spnq_train/QQQ",
    "VIXY": Path.home() / "train_data/spnq_train/VIXY",
}


def _apply_honest_env(*, deep_warmup: bool, frozen_norm: Path, slow_cfg: Path) -> None:
    """必须在 import baseline config / FCS 之前调用。"""
    os.environ["RUN_MODE"] = "BACKTEST"
    os.environ["REDIS_STREAM_SIM"] = "1"
    os.environ["QQQ_BTC_LIVE"] = "1"
    os.environ["QQQ_BTC_PUT_GATE_MODE"] = "vixy_z"
    os.environ["QQQ_BTC_REGIME_GOLD_1M"] = "0"
    os.environ.pop("QQQ_BTC_PUT_GATE_5M_FEATURE", None)
    os.environ.pop("GREEK_PARITY_MODE", None)
    os.environ.pop("FCS_MINUTE_PARITY_INJECT", None)
    os.environ["RECALC_GREEKS"] = "1"
    os.environ["FCS_FORCE_RECALC_GREEKS"] = "1"
    os.environ["FCS_DEBUG_RAW"] = "1"
    os.environ["FCS_OPTION_T_LABEL"] = os.environ.get("FCS_OPTION_T_LABEL", "end")
    os.environ["FCS_IV_PRICE_MODE"] = os.environ.get("FCS_IV_PRICE_MODE", "close")
    os.environ["FCS_TA_MONTH_ISOLATED"] = os.environ.get("FCS_TA_MONTH_ISOLATED", "1")
    os.environ["FCS_FROZEN_NORM_PATH"] = str(frozen_norm)
    os.environ["FCS_STATE_BACKEND"] = "none"
    os.environ["SLOW_FEATURE_CONFIG"] = str(slow_cfg)
    os.environ["SKIP_DEEP_WARMUP"] = "0" if deep_warmup else "1"
    os.environ.setdefault("FCS_NORMALIZER_STATS_UPDATE_INTERVAL", "1")
    # turbo 只在分钟边界 :00 调用 compute；grace=1 时 watermark 停在上一分钟，
    # 本分钟常无新 commit → assemble 空符号直接 None，下午漏采。grace=0 对齐秒级边界结算。
    os.environ["FCS_MINUTE_COMMIT_GRACE_SEC"] = os.environ.get("FCS_MINUTE_COMMIT_GRACE_SEC", "0")
    # 快路径不需要 persistence 写盘拖慢；仍保留 PG 连接供 warmup（若开启）
    os.environ.setdefault("SYNC_EXECUTION", "1")


def _setup_paths() -> None:
    for p in (str(_REPO), str(_BASELINE), str(_DAO)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _iso_to_ymd(date_iso: str) -> str:
    return date_iso.replace("-", "")


def _ymd_to_iso(ymd: str) -> str:
    ymd = ymd.replace("-", "")
    return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"


def _clear_fcs_redis_bars(r, symbols: list[str]) -> None:
    for sym in symbols:
        for key in (f"BAR:1M:{sym}", f"BAR_OPT:1M:{sym}"):
            try:
                r.delete(key)
            except Exception:
                pass


def _row_from_feat_list(
    *,
    ts: float,
    symbol: str,
    feat_names: list[str],
    vals: list[float],
) -> dict:
    row = {"ts": float(ts), "symbol": symbol}
    for name, val in zip(feat_names, vals):
        try:
            row[name] = float(val)
        except Exception:
            row[name] = np.nan
    return row


async def _replay_day(
    *,
    feat_svc,
    pitcher,
    ymd: str,
    symbol: str,
    collect_norm: bool,
    include_warmup: bool = False,
    progress_every: int = 1800,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from qqq_btc.tools.redis_fused_pitcher_1s import (  # noqa: WPS433
        _session_bounds_ts,
        _session_ts_list,
        set_replay_start_ts,
    )

    date_iso = _ymd_to_iso(ymd)
    t0 = time.time()
    map_b1, map_o1, map_b5, map_o5 = pitcher._load_day_maps(ymd)
    all_ts = _session_ts_list(ymd, include_preopen_minute=False)
    # 多喂 16:00:00，让 grace=1s 的 watermark 能结算 15:59 这根 bar
    _start_ts, end_ts = _session_bounds_ts(ymd, include_preopen_minute=False)
    if not all_ts or all_ts[-1] < end_ts:
        all_ts = list(all_ts) + [int(end_ts)]

    set_replay_start_ts(feat_svc.r, ymd, include_preopen_minute=False)
    _clear_fcs_redis_bars(feat_svc.r, list(feat_svc.symbols))

    last_known = {
        sym: {
            "ts": 0,
            "symbol": sym,
            "stock": {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0},
            "option_buckets": [],
            "option_contracts": [],
        }
        for sym in feat_svc.symbols
    }
    last_5m: dict[str, dict] = {}
    global_seq = 0
    raw_rows: list[dict] = []
    norm_rows: list[dict] = []
    first_full_min = ((all_ts[0] + 59) // 60) * 60 if all_ts else 0
    n_compute = 0

    for i, ts_val in enumerate(all_ts):
        ts_val = int(ts_val)
        feat_svc.r.set("replay:current_ts", str(ts_val))
        frame_complete = ts_val % 60 == 59
        minute_boundary = ts_val % 60 == 0
        frame_id = str(ts_val)

        b1_ts = map_b1.get(ts_val, {})
        o1_ts = map_o1.get(ts_val, {})
        b5_ts = map_b5.get(ts_val, {})
        o5_ts = map_o5.get(ts_val, {})

        batch_payloads = []
        for sym in feat_svc.symbols:
            payload = last_known[sym]
            payload["ts"] = ts_val
            global_seq += 1
            payload["frame_id"] = frame_id
            payload["seq"] = global_seq
            payload["frame_complete"] = frame_complete

            if sym in b1_ts:
                payload["stock"] = b1_ts[sym]
            elif payload["stock"]["close"] > 0:
                prev = float(payload["stock"]["close"])
                payload["stock"] = {
                    "open": prev,
                    "high": prev,
                    "low": prev,
                    "close": prev,
                    "volume": 0.0,
                }

            if sym in o1_ts:
                opt_data = o1_ts[sym]
                if isinstance(opt_data, dict):
                    payload["option_buckets"] = opt_data.get("buckets", [])
                    payload["option_contracts"] = opt_data.get("contracts", [])
                else:
                    payload["option_buckets"] = opt_data
                    payload["option_contracts"] = []

            if sym in b5_ts or sym in o5_ts:
                if sym not in last_5m:
                    last_5m[sym] = {}
                if sym in b5_ts:
                    last_5m[sym]["stock_5m"] = b5_ts[sym]
                if sym in o5_ts:
                    opt5 = o5_ts[sym]
                    if isinstance(opt5, dict):
                        last_5m[sym]["option_buckets_5m"] = opt5.get("buckets", [])
                        last_5m[sym]["option_contracts_5m"] = opt5.get("contracts", [])
                    else:
                        last_5m[sym]["option_buckets_5m"] = opt5
                        last_5m[sym]["option_contracts_5m"] = []

            if sym in last_5m:
                payload.update(last_5m[sym])

            last_known[sym] = payload
            # 浅拷贝顶层，避免后续秒覆盖本秒已入队字段
            batch_payloads.append(dict(payload))

        await feat_svc.process_market_data(batch_payloads)

        # 与 s2 turbo 一致：昂贵 compute 仅在分钟边界触发
        if (not minute_boundary) or ts_val < first_full_min:
            if progress_every and (i + 1) % progress_every == 0:
                print(
                    f"  [{date_iso}] tick {i+1}/{len(all_ts)} "
                    f"raw_rows={len(raw_rows)} compute={n_compute}",
                    flush=True,
                )
            continue

        feat_payload = await feat_svc.run_compute_cycle(
            ts_from_payload=ts_val, return_payload=True
        )
        n_compute += 1
        if not feat_payload or not bool(feat_payload.get("is_new_minute", False)):
            continue
        label_ts = float(feat_payload.get("ts") or ts_val)

        if not include_warmup:
            hist = feat_svc.committed_history_1min.get(symbol, pd.DataFrame())
            need = int(getattr(getattr(feat_svc, "market_profile", None), "warmup_required_len", 31) or 31)
            if hist is None or len(hist) < need:
                continue

        # 直接从 cached_batch_raw 抽目标标的（不依赖 payload.symbols，防 option gate 剔标的）
        batch_raw = getattr(feat_svc, "cached_batch_raw", None)
        try:
            b_idx = list(feat_svc.symbols).index(symbol)
        except ValueError:
            b_idx = -1
        if batch_raw is None or b_idx < 0 or b_idx >= len(batch_raw):
            continue
        raw_vec = batch_raw[b_idx]
        vals = []
        for fn in feat_svc.slow_feat_names:
            idx = feat_svc.feat_name_to_idx.get(fn)
            if idx is None or idx >= len(raw_vec):
                vals.append(0.0)
            else:
                try:
                    vals.append(float(raw_vec[idx]))
                except Exception:
                    vals.append(0.0)
        raw_rows.append(
            _row_from_feat_list(
                ts=label_ts,
                symbol=symbol,
                feat_names=list(feat_svc.slow_feat_names),
                vals=vals,
            )
        )

        if collect_norm:
            slow_list = feat_svc._build_slow_data_list_from_payload(feat_payload)
            for sym, vals in slow_list:
                if sym != symbol:
                    continue
                norm_rows.append(
                    _row_from_feat_list(
                        ts=label_ts,
                        symbol=sym,
                        feat_names=list(feat_svc.slow_feat_names),
                        vals=vals,
                    )
                )

        if progress_every and (i + 1) % progress_every == 0:
            print(
                f"  [{date_iso}] tick {i+1}/{len(all_ts)} "
                f"raw_rows={len(raw_rows)} compute={n_compute}",
                flush=True,
            )

    elapsed = time.time() - t0
    print(
        f"  [{date_iso}] done ticks={len(all_ts)} compute={n_compute} "
        f"raw={len(raw_rows)} norm={len(norm_rows)} in {elapsed:.1f}s",
        flush=True,
    )
    raw_df = pd.DataFrame(raw_rows) if raw_rows else pd.DataFrame()
    norm_df = pd.DataFrame(norm_rows) if norm_rows else pd.DataFrame()
    return raw_df, norm_df


async def _run_async(args: argparse.Namespace) -> int:
    _setup_paths()
    import baseline_paths  # noqa: E402,F401
    from config import REDIS_CFG, STREAM_FUSED_MARKET, STREAM_INFERENCE, TARGET_SYMBOLS  # noqa: E402
    from feature_compute_service_v8 import FeatureComputeService  # noqa: E402
    from qqq_btc.tools.compare_debug_slow_offline import (  # noqa: E402
        SKIP_FEATURES,
        _common_feat_cols,
        _load_offline,
        compare_day,
    )
    from qqq_btc.tools.redis_fused_pitcher_1s import RawParquetPitcher1s  # noqa: E402

    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    dates = [_ymd_to_iso(_iso_to_ymd(d)) for d in dates]
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    offline_raw = _load_offline(Path(args.offline_raw).expanduser(), dates)
    offline_norm = None
    if args.gate2:
        offline_norm = _load_offline(Path(args.offline_norm).expanduser(), dates)
        if offline_norm.empty:
            print("ERROR: offline quote_features_test empty")
            return 2
    if offline_raw.empty:
        print("ERROR: offline quote_features_raw empty")
        return 2

    pitcher = RawParquetPitcher1s(
        option_root=Path(args.option_root).expanduser(),
        stock_root=Path(args.stock_root).expanduser(),
        stock_fallback_roots=DEFAULT_STOCK_FALLBACKS,
        greek_root=Path(args.greek_root).expanduser(),
        greek_parity=False,
        symbols=list(TARGET_SYMBOLS),
        run_id="honest_gate1_fast",
    )

    slow_cfg = Path(os.environ["SLOW_FEATURE_CONFIG"])
    # New_Pro/CONFIG 与 baseline_qqq/CONFIG 两处都可能有 fast 配置
    fast_cfg = _REPO / "New_Pro" / "CONFIG" / "fast_feature_qqq.json"
    if not fast_cfg.exists():
        fast_cfg = _BASELINE / "CONFIG" / "fast_feature_qqq.json"
    if not fast_cfg.exists():
        fast_cfg = _REPO / "New_Pro" / "CONFIG" / "fast_feature.json"
    config_paths = {"fast": str(fast_cfg), "slow": str(slow_cfg)}

    feat_cfg = copy.deepcopy(REDIS_CFG)
    feat_cfg["input_stream"] = STREAM_FUSED_MARKET
    feat_cfg["raw_stream"] = STREAM_FUSED_MARKET
    feat_cfg["output_stream"] = STREAM_INFERENCE

    print(
        f"[fast] symbols={TARGET_SYMBOLS} dates={dates} "
        f"SKIP_DEEP_WARMUP={os.environ.get('SKIP_DEEP_WARMUP')} "
        f"FCS_DEBUG_RAW={os.environ.get('FCS_DEBUG_RAW')} "
        f"greek_parity=False",
        flush=True,
    )
    feat_svc = FeatureComputeService(feat_cfg, list(TARGET_SYMBOLS), config_paths)
    if not getattr(feat_svc, "debug_raw_enabled", False):
        print("ERROR: FCS_DEBUG_RAW not enabled on FeatureComputeService")
        return 2

    if args.deep_warmup:
        print("[fast] running deep warmup from PG...", flush=True)
        feat_svc._robust_backfill_and_warmup()

    live_raw_frames: list[pd.DataFrame] = []
    live_norm_frames: list[pd.DataFrame] = []
    is_first = True
    for date_iso in dates:
        ymd = _iso_to_ymd(date_iso)
        print(f"\n======== FAST {date_iso} ========", flush=True)
        if is_first:
            if hasattr(feat_svc, "reset_internal_memory"):
                # deep warmup 已灌入时不要抹掉；无 warmup 则冷启动
                if not args.deep_warmup:
                    feat_svc.reset_internal_memory()
            print("[fast] cold start", flush=True)
        else:
            # 价格 hist 热启动保覆盖；期权动量/IV 状态按日隔离（对齐离线 pct_change fillna(0)）
            from collections import deque

            for sym in list(getattr(feat_svc, "symbols", []) or []):
                if hasattr(feat_svc, "deriv_history"):
                    feat_svc.deriv_history[sym] = deque(maxlen=10)
            eng = getattr(getattr(feat_svc, "engine_adapter", None), "engine", None)
            if eng is not None and hasattr(eng, "_last_good_iv"):
                eng._last_good_iv.clear()
            print("[fast] hot start prices; reset option deriv/IV state", flush=True)

        raw_df, norm_df = await _replay_day(
            feat_svc=feat_svc,
            pitcher=pitcher,
            ymd=ymd,
            symbol=args.symbol,
            collect_norm=bool(args.gate2),
            include_warmup=bool(args.include_warmup),
            progress_every=args.progress_every,
        )
        if not raw_df.empty:
            live_raw_frames.append(raw_df)
            raw_path = out_dir / f"live_raw_{ymd}.parquet"
            raw_df.to_parquet(raw_path, index=False)
            print(f"  wrote {raw_path}", flush=True)
        if args.gate2 and not norm_df.empty:
            live_norm_frames.append(norm_df)
            norm_path = out_dir / f"live_norm_{ymd}.parquet"
            norm_df.to_parquet(norm_path, index=False)
        is_first = False

    if not live_raw_frames:
        print("ERROR: no live raw feature rows produced")
        return 2

    live_raw = pd.concat(live_raw_frames, ignore_index=True)
    feats = _common_feat_cols(offline_raw, live_raw)
    _ = SKIP_FEATURES
    print(
        f"\n[Gate-1 RAW FAST] offline={len(offline_raw)} live={len(live_raw)} "
        f"common_feats={len(feats)} ts_shift_sec={args.ts_shift_sec} "
        f"med_tol={args.med_tol} corr_min={args.corr_min}",
        flush=True,
    )

    by_day = []
    for d in dates:
        rep = compare_day(
            offline_raw,
            live_raw,
            date=d,
            feats=feats,
            med_tol=args.med_tol,
            corr_min=args.corr_min,
            ts_shift_sec=args.ts_shift_sec,
        )
        by_day.append(rep)
        status = "PASS" if rep.get("pass") else "FAIL"
        print(
            f"\n=== {d} [{status}] matched={rep.get('n_matched')} "
            f"pass_rate={rep.get('pass_rate', 0):.1%} ===",
            flush=True,
        )
        if rep.get("reason"):
            print(f"  reason: {rep['reason']}", flush=True)
        for w in rep.get("worst") or []:
            if w.get("pass"):
                continue
            print(
                f"  FAIL {w['feature']:28s} med={w.get('med_abs_err')} "
                f"max={w.get('max_abs_err')} corr={w.get('corr')}",
                flush=True,
            )
        fails = rep.get("failed_features") or []
        if fails:
            print(f"  failed({len(fails)}): {fails[:12]}", flush=True)

    overall = all(r.get("pass") for r in by_day)
    summary = {
        "gate": 1,
        "name": "fcs_fast_raw_vs_quote_features_raw",
        "mode": "honest_gate1_fcs_fast",
        "offline": str(Path(args.offline_raw).expanduser()),
        "option_root": str(Path(args.option_root).expanduser()),
        "greek_root": str(Path(args.greek_root).expanduser()),
        "symbol": args.symbol,
        "med_tol": args.med_tol,
        "corr_min": args.corr_min,
        "ts_shift_sec": args.ts_shift_sec,
        "n_feats": len(feats),
        "overall_pass": overall,
        "by_day": [{k: v for k, v in r.items() if k != "columns"} for r in by_day],
        "env": {
            "FCS_DEBUG_RAW": os.environ.get("FCS_DEBUG_RAW"),
            "RECALC_GREEKS": os.environ.get("RECALC_GREEKS"),
            "FCS_FORCE_RECALC_GREEKS": os.environ.get("FCS_FORCE_RECALC_GREEKS"),
            "FCS_TA_MONTH_ISOLATED": os.environ.get("FCS_TA_MONTH_ISOLATED"),
            "FCS_OPTION_T_LABEL": os.environ.get("FCS_OPTION_T_LABEL"),
            "FCS_IV_PRICE_MODE": os.environ.get("FCS_IV_PRICE_MODE"),
            "SKIP_DEEP_WARMUP": os.environ.get("SKIP_DEEP_WARMUP"),
            "greek_parity": False,
        },
        "next": "Gate-2 only if overall_pass",
    }
    out1 = out_dir / "feat_parity_gate1_raw.json"
    full = dict(summary)
    full["by_day_full"] = by_day
    out1.write_text(json.dumps(full, indent=2, ensure_ascii=False, default=str))
    print(f"\n=== GATE-1 RAW OVERALL: {'PASS' if overall else 'FAIL'} ===", flush=True)
    print(f"wrote {out1}", flush=True)

    if args.gate2 and overall and live_norm_frames and offline_norm is not None:
        live_norm = pd.concat(live_norm_frames, ignore_index=True)
        feats2 = _common_feat_cols(offline_norm, live_norm)
        print(
            f"\n[Gate-2 NORM FAST] offline={len(offline_norm)} live={len(live_norm)} "
            f"common_feats={len(feats2)} med_tol={args.gate2_med_tol} "
            f"corr_min={args.corr_min} ts_shift_sec={args.ts_shift_sec}",
            flush=True,
        )
        by_day2 = []
        for d in dates:
            rep = compare_day(
                offline_norm,
                live_norm,
                date=d,
                feats=feats2,
                med_tol=args.gate2_med_tol,
                corr_min=args.corr_min,
                ts_shift_sec=args.ts_shift_sec,
            )
            by_day2.append(rep)
            status = "PASS" if rep.get("pass") else "FAIL"
            print(
                f"\n=== Gate2 {d} [{status}] matched={rep.get('n_matched')} "
                f"pass_rate={rep.get('pass_rate', 0):.1%} ===",
                flush=True,
            )
            if rep.get("reason"):
                print(f"  reason: {rep['reason']}", flush=True)
            for w in rep.get("worst") or []:
                if w.get("pass"):
                    continue
                print(
                    f"  FAIL {w['feature']:28s} med={w.get('med_abs_err')} "
                    f"max={w.get('max_abs_err')} corr={w.get('corr')}",
                    flush=True,
                )
            fails = rep.get("failed_features") or []
            if fails:
                print(f"  failed({len(fails)}): {fails[:12]}", flush=True)
        overall2 = all(r.get("pass") for r in by_day2)
        summary2 = {
            "gate": 2,
            "name": "fcs_fast_norm_vs_quote_features_test",
            "overall_pass": overall2,
            "med_tol": args.gate2_med_tol,
            "corr_min": args.corr_min,
            "by_day": [{k: v for k, v in r.items() if k != "columns"} for r in by_day2],
        }
        out2 = out_dir / "feat_parity_gate2_norm.json"
        out2.write_text(
            json.dumps(
                {**summary2, "by_day_full": by_day2},
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
        print(f"=== GATE-2 NORM OVERALL: {'PASS' if overall2 else 'FAIL'} ===", flush=True)
        print(f"wrote {out2}", flush=True)
        if not overall2:
            return 2
    elif args.gate2 and not overall:
        print("[Gate-2] skipped because Gate-1 FAIL", flush=True)

    return 0 if overall else 2


def main() -> int:
    ap = argparse.ArgumentParser(description="Honest Gate1/2 in-process FCS fast path")
    ap.add_argument("--dates", default=DEFAULT_DATES)
    ap.add_argument("--option-root", default=str(DEFAULT_OPT_ROOT))
    ap.add_argument("--greek-root", default=str(DEFAULT_GREEK))
    ap.add_argument("--stock-root", default=str(DEFAULT_STOCK_ROOT))
    ap.add_argument("--offline-raw", default=str(DEFAULT_OFFLINE_RAW))
    ap.add_argument("--offline-norm", default=str(DEFAULT_OFFLINE_NORM))
    ap.add_argument(
        "--out",
        default=str(_REPO / "qqq_btc/results/july_w1_ft56_honest_gate1_fast"),
    )
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--med-tol", type=float, default=1e-3)
    ap.add_argument("--corr-min", type=float, default=0.90)
    ap.add_argument("--ts-shift-sec", type=int, default=60)
    ap.add_argument("--gate2", action="store_true", help="Also compare norm features (Gate2)")
    ap.add_argument("--gate2-med-tol", type=float, default=0.05)
    ap.add_argument(
        "--deep-warmup",
        action="store_true",
        help="PG Deep Warmup (SKIP_DEEP_WARMUP=0); default skip for speed",
    )
    ap.add_argument(
        "--include-warmup",
        action="store_true",
        help="Also collect pre-warmup minutes (raw often 0; hurts corr)",
    )
    ap.add_argument("--progress-every", type=int, default=1800)
    ap.add_argument(
        "--frozen-norm",
        default=str(_REPO / "qqq_btc/CONFIG/frozen_norm_qqq_daily.npz"),
    )
    ap.add_argument(
        "--slow-config",
        default=str(_REPO / "qqq_btc/CONFIG/slow_feature_qqq_v4.json"),
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [GATE1_FAST] - %(message)s",
    )

    frozen = Path(args.frozen_norm).expanduser()
    slow_cfg = Path(args.slow_config).expanduser()
    if not frozen.exists():
        print(f"ERROR: frozen norm missing: {frozen}")
        return 2
    if not slow_cfg.exists():
        print(f"ERROR: slow config missing: {slow_cfg}")
        return 2

    _apply_honest_env(
        deep_warmup=bool(args.deep_warmup),
        frozen_norm=frozen,
        slow_cfg=slow_cfg,
    )
    return asyncio.run(_run_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
