#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polygon / Massive 秒级期权 Quote 下载器（1DTE old-lock 口径）

默认锁约表由 step1_build_target_map_old.py（trading-1DTE）生成，
不要再用 step1_build_target_map.py.duplicate 的开盘窗锁约去下载。

上游锁约（先跑）:
  cd preprocess/download
  python step1_build_target_map_old.py \\
      --dte-mode trading \\
      --config ../CONFIG/anchor_qqq_1dte_4bucket.json \\
      --start-date 2026-04-01 --end-date 2026-07-10 \\
      --raw-dir ~/train_data/nq_options_day_iv \\
      --output ~/train_data/locked_targets_map_old_style_trading_1dte.parquet

本脚本只下载 1s bid/ask quote + bucket metadata，不在下载阶段算 Greeks。
分钟级 IV/Greeks 由后续链路统一计算:

  step3 1s→1m → option_cac_day_vectorized_day.py → options_locked_feature

用法:
  export MASSIVE_API_KEY=...   # 或 POLYGON_API_KEY
  python step2_polygon_second_sniper_v1.py
  python step2_polygon_second_sniper_v1.py --start-date 2026-06-01 --end-date 2026-06-30
  python step2_polygon_second_sniper_v1.py --force --no-download-stock
  python step2_polygon_second_sniper_v1.py --compute-greeks   # 仅调试/秒级研究

默认会在下期权前做 Stock 1s preflight：锁约表里每个 (symbol, date) 若秒级正股
缺失/过薄，先批量补齐，再并行下合约 quote（避免左标签 1m ffill 冒充秒级）。

正股 1s 下载逻辑见同目录 ``download_stock_1s.py``（可单独跑）。
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import logging
import os
import re
from pytz import timezone

import numpy as np
import pandas as pd
from polygon import RESTClient
from tqdm import tqdm

try:
    from preprocess.download.download_stock_1s import (
        DEFAULT_STOCK_OUTPUT_DIR,
        MIN_STOCK_1S_ROWS,
        download_stock_1s_day as _download_stock_1s_day,
        ensure_stock_1s_for_map as _ensure_stock_1s_for_map,
        load_stock_price_map as _load_stock_price_map,
        resolve_api_key as _resolve_api_key,
        stock_1s_file_ok as _stock_1s_file_ok,
        stock_1s_path as _stock_1s_path,
    )
except ImportError:  # `cd preprocess/download && python step2_...py`
    from download_stock_1s import (  # type: ignore
        DEFAULT_STOCK_OUTPUT_DIR,
        MIN_STOCK_1S_ROWS,
        download_stock_1s_day as _download_stock_1s_day,
        ensure_stock_1s_for_map as _ensure_stock_1s_for_map,
        load_stock_price_map as _load_stock_price_map,
        resolve_api_key as _resolve_api_key,
        stock_1s_file_ok as _stock_1s_file_ok,
        stock_1s_path as _stock_1s_path,
    )

# ================= 全局配置 =================
API_KEY = None  # main() 里再解析，避免 import 时强依赖 key
TARGET_MAP_FILE = os.path.expanduser(
    "~/train_data/locked_targets_map_old_style_trading_1dte.parquet"
)
OUTPUT_DIR = "/mnt/s990/data/raw_1s/dte1_options_old_lock"
STOCK_OUTPUT_DIR = DEFAULT_STOCK_OUTPUT_DIR
RFR_CACHE_FILE = os.path.expanduser("~/risk_free_rates.parquet")

MAX_WORKERS = 50
DOWNLOAD_OPTIONS = True
# 默认 False：训练用 Greeks 在 1min 层 (option_cac_day_vectorized_day) 重算
COMPUTE_GREEKS_AT_1S = False
# 默认 True：保留 1s 正股供 L2 event replay；若只要 quote 可关
DOWNLOAD_STOCK_1S = True
FORCE_OVERWRITE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("urllib3").setLevel(logging.ERROR)
logger = logging.getLogger("Polygon_1s_Sniper")
eastern = timezone("America/New_York")

QUOTE_OUTPUT_COLS = [
    "ts",
    "timestamp",
    "ticker",
    "tag",
    "bucket_id",
    "underlying",
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "price",
    "mid_price",
    "strike",
]

GREEKS_COLS = ["iv", "delta", "gamma", "vega", "theta"]

try:
    from py_vollib_vectorized import get_all_greeks, vectorized_implied_volatility

    HAS_VOLLIB = True
except ImportError:
    HAS_VOLLIB = False
    logger.warning("py_vollib_vectorized not found; --compute-greeks will write zeros.")


def compute_greeks_for_df(df: pd.DataFrame, stock_price_map: dict, r_val: float = 0.04) -> pd.DataFrame:
    """可选：秒级 BSM IV + Greeks（默认不在 sniper 阶段调用）。"""
    if not HAS_VOLLIB or df.empty:
        for col in GREEKS_COLS:
            if col not in df.columns:
                df[col] = 0.0
        return df

    work = df.copy()
    clean_tickers = work["ticker"].str.replace("O:", "", regex=False)
    extracted = clean_tickers.str.extract(r"^[A-Z]+(\d{6})([CP])\d{8}$")
    work["expiry"] = pd.to_datetime("20" + extracted[0], format="%Y%m%d", errors="coerce")
    work["opt_type"] = extracted[1].map({"C": "c", "P": "p"})

    stk_ts = work["ts"].round(0).astype("int64")
    work["stock_close"] = stk_ts.map(stock_price_map)
    work["stock_close"] = work["stock_close"].ffill().bfill()

    current_ts = work["timestamp"]
    expiry_ts = work["expiry"].dt.tz_localize("America/New_York") + pd.Timedelta(hours=16)
    t_years = (expiry_ts - current_ts).dt.total_seconds().values / 31557600.0
    t_years = np.maximum(t_years, 1e-6)

    p = work["price"].values.astype(float)
    s = work["stock_close"].values.astype(float)
    k = work["strike"].values.astype(float)
    r = np.full_like(p, r_val)
    is_call = (work["opt_type"] == "c").values
    is_put = (work["opt_type"] == "p").values

    iv = np.zeros_like(p, dtype=float)
    valid = (p > 0.0) & (s > 0.0) & (k > 0.0) & (t_years > 0.0)
    try:
        if (is_call & valid).any():
            m = is_call & valid
            iv[m] = vectorized_implied_volatility(
                p[m], s[m], k[m], t_years[m], r[m], "c", return_as="numpy", on_error="ignore"
            )
        if (is_put & valid).any():
            m = is_put & valid
            iv[m] = vectorized_implied_volatility(
                p[m], s[m], k[m], t_years[m], r[m], "p", return_as="numpy", on_error="ignore"
            )
    except Exception as exc:
        logger.debug("vectorized_implied_volatility error: %s", exc)

    iv = np.nan_to_num(iv, nan=0.0)
    work["iv"] = iv

    delta = np.zeros_like(iv)
    gamma = np.zeros_like(iv)
    vega = np.zeros_like(iv)
    theta = np.zeros_like(iv)
    valid_iv_mask = (iv > 0.01) & (iv < 5.0) & valid
    try:
        if is_call.any() and valid_iv_mask[is_call].any():
            vc = is_call & valid_iv_mask
            g_df = get_all_greeks("c", s[vc], k[vc], t_years[vc], r[vc], iv[vc], return_as="dataframe")
            delta[vc] = g_df["delta"].values
            gamma[vc] = g_df["gamma"].values
            vega[vc] = g_df["vega"].values
            theta[vc] = g_df["theta"].values
        if is_put.any() and valid_iv_mask[is_put].any():
            vp = is_put & valid_iv_mask
            g_df = get_all_greeks("p", s[vp], k[vp], t_years[vp], r[vp], iv[vp], return_as="dataframe")
            delta[vp] = g_df["delta"].values
            gamma[vp] = g_df["gamma"].values
            vega[vp] = g_df["vega"].values
            theta[vp] = g_df["theta"].values
    except Exception as exc:
        logger.info("get_all_greeks failed: %s", exc)

    work["delta"], work["gamma"], work["vega"], work["theta"] = delta, gamma, vega, theta
    work.drop(columns=["expiry", "opt_type", "stock_close"], inplace=True, errors="ignore")
    return work


def stock_1s_path(symbol: str, date_str: str) -> str:
    return _stock_1s_path(symbol, date_str, stock_output_dir=STOCK_OUTPUT_DIR)


def stock_1s_file_ok(path: str, *, min_rows: int = MIN_STOCK_1S_ROWS) -> bool:
    return _stock_1s_file_ok(path, min_rows=min_rows)


def download_stock_1s_day(client: RESTClient, symbol: str, date_str: str) -> bool:
    return _download_stock_1s_day(client, symbol, date_str, stock_output_dir=STOCK_OUTPUT_DIR)


def ensure_stock_1s_for_map(target_map: pd.DataFrame, *, max_workers: int = 12) -> dict[str, int]:
    """Before option quotes: download every missing/thin 1s underlying in the map."""
    return _ensure_stock_1s_for_map(
        target_map,
        stock_output_dir=STOCK_OUTPUT_DIR,
        max_workers=max_workers,
        api_key=API_KEY,
    )


def load_stock_price_map(client: RESTClient, symbol: str, date_str: str) -> dict:
    """加载 1s 正股 close（缺则先下）。不再用 1m ffill 冒充秒级。"""
    return _load_stock_price_map(client, symbol, date_str, stock_output_dir=STOCK_OUTPUT_DIR)

def download_option_quotes(
    client: RESTClient,
    symbol: str,
    date_str: str,
    group_df: pd.DataFrame,
    *,
    window_start: str = "09:30",
    window_end: str = "16:00",
    contract_workers: int = 1,
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    gte_ns, lte_ns = _rth_ns_bounds(date_str, window_start=window_start, window_end=window_end)
    t_start = datetime.datetime.strptime(window_start, "%H:%M").time()
    t_end = datetime.datetime.strptime(window_end, "%H:%M").time()

    def _one(row) -> tuple[pd.DataFrame | None, str | None]:
        occ = row["contract_symbol"]
        poly_ticker = occ if str(occ).startswith("O:") else f"O:{occ}"
        try:
            # Stream pages; floor to 1s on the fly to avoid holding full tick buffer.
            last_by_sec: dict[int, dict] = {}
            for q in client.list_quotes(
                ticker=poly_ticker,
                timestamp_gte=gte_ns,
                timestamp_lte=lte_ns,
                limit=50000,
            ):
                ts_ns = getattr(q, "sip_timestamp", None) or getattr(q, "participant_timestamp", 0)
                bid = float(getattr(q, "bid_price", 0.0) or 0.0)
                ask = float(getattr(q, "ask_price", 0.0) or 0.0)
                if bid <= 0 or ask < bid:
                    continue
                sec = int(ts_ns // 1_000_000_000)
                last_by_sec[sec] = {
                    "ts_ns": ts_ns,
                    "bid": bid,
                    "ask": ask,
                    "bid_size": float(getattr(q, "bid_size", 0.0) or 0.0),
                    "ask_size": float(getattr(q, "ask_size", 0.0) or 0.0),
                }
            if not last_by_sec:
                return None, str(occ)
            rows = sorted(last_by_sec.items(), key=lambda x: x[0])
            df = pd.DataFrame([r for _, r in rows])
            df["timestamp"] = pd.to_datetime(df["ts_ns"], unit="ns", utc=True).dt.tz_convert(eastern)
            time_series = df["timestamp"].dt.time
            df = df[(time_series >= t_start) & (time_series < t_end)].copy()
            if df.empty:
                return None, str(occ)
            df["ts_1s"] = df["timestamp"].dt.floor("1s")
            df = df.drop_duplicates(subset=["ts_1s"], keep="last").copy()
            df["timestamp"] = df["ts_1s"]
            df["ts"] = df["timestamp"].astype("int64") / 1e9
            df["ticker"] = str(occ).replace("O:", "")
            df["bucket_id"] = row["bucket_id"]
            df["tag"] = row.get("tag", "")
            df["underlying"] = symbol
            df["mid_price"] = (df["bid"] + df["ask"]) / 2.0
            df["price"] = df["mid_price"]
            match = re.search(r"[CP](\d{8})$", df["ticker"].iloc[0])
            df["strike"] = float(match.group(1)) / 1000.0 if match else 0.0
            return df, None
        except Exception as exc:
            logger.error("Contract %s process error: %s", occ, exc, exc_info=True)
            return None, str(occ)

    rows = [r for _, r in group_df.iterrows()]
    n_cw = max(1, int(contract_workers))
    if n_cw == 1 or len(rows) <= 1:
        results = [_one(r) for r in rows]
    else:
        # Intra-day contract parallelism (I/O bound). Nested under day ThreadPool —
        # ProcessPool per symbol does NOT help: CPU~5%, bottleneck is quote pagination.
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(n_cw, len(rows))) as pool:
            results = list(pool.map(_one, rows))

    for df, miss in results:
        if df is not None:
            frames.append(df)
        if miss is not None:
            missing.append(miss)
    if missing:
        logger.warning("%s %s: no usable quotes for %d/%d contracts: %s",
                       symbol, date_str, len(missing), len(group_df), missing)
    return frames


def _rth_ns_bounds(
    date_str: str,
    *,
    window_start: str = "09:30",
    window_end: str = "16:00",
) -> tuple[int, int]:
    """NYSE window as SIP nanosecond bounds."""
    start = eastern.localize(
        datetime.datetime.strptime(f"{date_str} {window_start}:00", "%Y-%m-%d %H:%M:%S")
    )
    end = eastern.localize(
        datetime.datetime.strptime(f"{date_str} {window_end}:00", "%Y-%m-%d %H:%M:%S")
    )
    return int(start.timestamp() * 1e9), int(end.timestamp() * 1e9)


def process_single_day_polygon(args):
    (
        symbol,
        date_str,
        group_df,
        r_val,
        compute_greeks,
        download_stock,
        allow_partial,
        window_start,
        window_end,
        contract_workers,
    ) = args
    client = RESTClient(API_KEY)

    out_dir = os.path.join(OUTPUT_DIR, symbol)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{symbol}_{date_str}.parquet")

    if os.path.exists(out_path) and not FORCE_OVERWRITE:
        return f"⏩ {symbol} {date_str} exists"

    need_stock = download_stock or compute_greeks
    stock_price_map = load_stock_price_map(client, symbol, date_str) if need_stock else {}

    if not DOWNLOAD_OPTIONS and os.path.exists(out_path):
        try:
            final_df = pd.read_parquet(out_path)
            if compute_greeks:
                final_df = final_df.drop(columns=[c for c in GREEKS_COLS if c in final_df.columns])
            else:
                return f"⏩ {symbol} {date_str} exists (quotes-only, no recompute)"
        except Exception as exc:
            logger.debug("Failed to load existing option data for %s %s: %s", symbol, date_str, exc)
            final_df = None
    else:
        final_df = None

    if final_df is None:
        quote_frames = (
            download_option_quotes(
                client,
                symbol,
                date_str,
                group_df,
                window_start=window_start,
                window_end=window_end,
                contract_workers=contract_workers,
            )
            if DOWNLOAD_OPTIONS
            else []
        )
        if not quote_frames:
            return f"⚠️ {symbol} {date_str}: No valid quotes."
        final_df = pd.concat(quote_frames, ignore_index=True)
        n_locked = int(group_df["bucket_id"].nunique()) if "bucket_id" in group_df.columns else len(group_df)
        n_got = int(final_df["bucket_id"].nunique()) if "bucket_id" in final_df.columns else final_df["ticker"].nunique()
        if n_got < n_locked and not allow_partial:
            logger.warning(
                "%s %s: incomplete buckets got=%d locked=%d tickers=%s — skip write",
                symbol,
                date_str,
                n_got,
                n_locked,
                sorted(final_df["ticker"].astype(str).unique()),
            )
            return (
                f"⚠️ {symbol} {date_str}: incomplete {n_got}/{n_locked} buckets, not written"
            )
        if n_got < n_locked and allow_partial:
            logger.warning(
                "%s %s: partial buckets got=%d locked=%d — writing anyway (--allow-partial)",
                symbol,
                date_str,
                n_got,
                n_locked,
            )

    if compute_greeks:
        if not stock_price_map:
            logger.warning("%s %s: Greeks requested but stock map empty; writing zero Greeks.", symbol, date_str)
        final_df = compute_greeks_for_df(final_df, stock_price_map, r_val=r_val)
        out_cols = QUOTE_OUTPUT_COLS + GREEKS_COLS
    else:
        out_cols = QUOTE_OUTPUT_COLS

    final_df = final_df[[c for c in out_cols if c in final_df.columns]]
    final_df.to_parquet(out_path, engine="pyarrow", index=False, compression="zstd")
    mode = "quotes+greeks" if compute_greeks else "quotes-only"
    return f"🎯 {symbol} {date_str}: {mode} | {len(final_df)} rows | tickers={final_df['ticker'].nunique()}"


def load_rfr_series():
    try:
        import pandas_datareader.data as web

        fetch_start = pd.Timestamp(datetime.date(2020, 1, 1)).normalize()
        fetch_end = pd.Timestamp(datetime.date.today() + datetime.timedelta(days=14)).normalize()
        logger.info("Downloading fresh RFR data from FRED...")
        new_data = web.DataReader("DGS3MO", "fred", fetch_start, fetch_end)
        new_data.index = pd.to_datetime(new_data.index).normalize()
        new_data = new_data / 100.0
        rfr_df = new_data.resample("D").ffill().fillna(0.04)
        try:
            rfr_df.to_parquet(RFR_CACHE_FILE)
        except Exception as exc:
            logger.warning("Could not save RFR cache: %s", exc)
        return rfr_df["DGS3MO"]
    except Exception as exc:
        logger.warning("Failed to download RFR: %s. Falling back to cache.", exc)
        try:
            if os.path.exists(RFR_CACHE_FILE):
                df = pd.read_parquet(RFR_CACHE_FILE)
                if isinstance(df, pd.DataFrame):
                    return df.iloc[:, 0]
                return df
        except Exception as cache_exc:
            logger.error("Cache load also failed: %s", cache_exc)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Polygon/Massive 1s option quote sniper (old-style trading-1DTE lock map)"
    )
    parser.add_argument(
        "--target-map",
        default=TARGET_MAP_FILE,
        help="locked map from step1_build_target_map_old.py",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--stock-output-dir", default=STOCK_OUTPUT_DIR)
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD inclusive")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD inclusive")
    parser.add_argument("--symbols", default=None, help="comma-separated, default=all in map")
    parser.add_argument("--compute-greeks", action="store_true", help="optional 1s BSM Greeks (default off)")
    parser.add_argument(
        "--download-stock",
        action="store_true",
        default=None,
        help="preflight+cache 1s underlyings before option quotes (default on)",
    )
    parser.add_argument(
        "--no-download-stock",
        action="store_true",
        help="skip 1s stock preflight (not recommended)",
    )
    parser.add_argument(
        "--stock-workers",
        type=int,
        default=12,
        help="parallel workers for stock 1s preflight (default 12)",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing day files")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="write day file even if some locked contracts have no quotes (research maps)",
    )
    parser.add_argument(
        "--window-start",
        default="10:00",
        help="quote window start HH:MM ET (default 10:00; Mag7 signals start 10:30)",
    )
    parser.add_argument(
        "--window-end",
        default="15:00",
        help="quote window end HH:MM ET (default 15:00; hold<=30m after 14:00)",
    )
    parser.add_argument(
        "--global-pool",
        action="store_true",
        default=True,
        help="parallelize across all symbol-days (default on; much faster than per-symbol)",
    )
    parser.add_argument("--no-global-pool", action="store_true", help="legacy: one symbol at a time")
    parser.add_argument(
        "--contract-workers",
        type=int,
        default=4,
        help="parallel quote streams per day-task (default 4; miss map ~3 contracts/day)",
    )
    return parser.parse_args()


def main() -> None:
    global API_KEY, OUTPUT_DIR, STOCK_OUTPUT_DIR, FORCE_OVERWRITE
    args = parse_args()
    API_KEY = _resolve_api_key()
    OUTPUT_DIR = args.output_dir
    STOCK_OUTPUT_DIR = args.stock_output_dir
    FORCE_OVERWRITE = args.force
    use_global = bool(args.global_pool) and not bool(args.no_global_pool)
    window_start = str(args.window_start)
    window_end = str(args.window_end)

    compute_greeks = args.compute_greeks or COMPUTE_GREEKS_AT_1S
    if args.no_download_stock:
        download_stock = False
    elif args.download_stock is True:
        download_stock = True
    else:
        download_stock = DOWNLOAD_STOCK_1S

    if not os.path.exists(args.target_map):
        logger.error("Target map not found: %s", args.target_map)
        logger.error(
            "先运行: python step1_build_target_map_old.py --dte-mode trading "
            "--config ../CONFIG/anchor_qqq_1dte_4bucket.json ..."
        )
        return

    target_map = pd.read_parquet(args.target_map)
    if "date_str" not in target_map.columns or "contract_symbol" not in target_map.columns:
        logger.error("target map missing date_str/contract_symbol: %s", list(target_map.columns))
        return
    target_map["date_str"] = target_map["date_str"].astype(str)
    if args.start_date:
        target_map = target_map[target_map["date_str"] >= args.start_date]
    if args.end_date:
        target_map = target_map[target_map["date_str"] <= args.end_date]
    if args.symbols:
        want = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        target_map = target_map[target_map["symbol"].astype(str).str.upper().isin(want)]
    if target_map.empty:
        logger.error("no rows left after filters")
        return

    rfr_series = load_rfr_series() if compute_greeks else None

    symbols = target_map["symbol"].unique()
    logger.info(
        "Polygon 1s sniper | map=%s | days=%d | symbols=%d | compute_greeks=%s | download_stock=%s | allow_partial=%s | global_pool=%s | day_workers=%d | contract_workers=%d | window=%s-%s | out=%s",
        args.target_map,
        target_map["date_str"].nunique(),
        len(symbols),
        compute_greeks,
        download_stock,
        args.allow_partial,
        use_global,
        args.max_workers,
        max(1, int(args.contract_workers)),
        window_start,
        window_end,
        OUTPUT_DIR,
    )
    if "front_dte" in target_map.columns:
        logger.info("front_dte dist:\n%s", target_map.groupby("date_str")["front_dte"].first().value_counts().sort_index())

    # Always preflight 1s stocks for the filtered map before any option day-tasks.
    if download_stock:
        ensure_stock_1s_for_map(target_map, max_workers=max(1, int(args.stock_workers)))
    else:
        logger.warning("Stock 1s preflight skipped (--no-download-stock)")

    def _task(sym: str, d: str, g: pd.DataFrame):
        r_val = 0.045
        if rfr_series is not None:
            try:
                t_date = pd.to_datetime(d).normalize()
                if t_date in rfr_series.index:
                    r_val = float(rfr_series.loc[t_date])
                else:
                    idx = rfr_series.index.searchsorted(t_date.tz_localize(None))
                    idx = np.clip(idx, 0, len(rfr_series) - 1)
                    r_val = float(rfr_series.iloc[idx])
            except Exception as exc:
                logger.debug("RFR lookup error for %s: %s", d, exc)
        return (
            sym,
            d,
            g,
            r_val,
            compute_greeks,
            download_stock,
            bool(args.allow_partial),
            window_start,
            window_end,
            max(1, int(args.contract_workers)),
        )

    all_tasks = []
    for sym in symbols:
        sym_df = target_map[target_map["symbol"] == sym]
        for d, g in sym_df.groupby("date_str"):
            out_path = os.path.join(OUTPUT_DIR, sym, f"{sym}_{d}.parquet")
            if os.path.exists(out_path) and not FORCE_OVERWRITE:
                continue
            all_tasks.append(_task(sym, d, g))

    if not all_tasks:
        logger.info("Nothing to download (all day files exist).")
        return

    logger.info("Dispatching %d day-tasks with max_workers=%d", len(all_tasks), args.max_workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        list(
            tqdm(
                executor.map(process_single_day_polygon, all_tasks),
                total=len(all_tasks),
                desc="All symbols",
            )
        )

    logger.info("All symbols processed.")


if __name__ == "__main__":
    main()
