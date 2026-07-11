#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polygon 秒级期权 Quote 下载器（与 Databento sniper 对齐）

默认只下载 1s bid/ask quote + bucket metadata，不在下载阶段计算 Greeks。
分钟级 IV/Greeks 由后续链路统一计算：

  step3 聚合 1min → option_cac_day_vectorized_day.py → options_locked_feature

用法:
  python step2_polygon_second_sniper_v1.py
  python step2_polygon_second_sniper_v1.py --compute-greeks   # 仅调试/秒级研究
  python step2_polygon_second_sniper_v1.py --force
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

# ================= 全局配置 =================
API_KEY = os.environ.get("POLYGON_API_KEY", "JXuIcG_dpoRiCE6jP7c73nVWweEVSpUp")
TARGET_MAP_FILE = os.path.expanduser("~/train_data/locked_targets_map.parquet")
OUTPUT_DIR = "/mnt/s990/data/raw_1s/options"
STOCK_OUTPUT_DIR = "/mnt/s990/data/raw_1s/stocks"
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


def load_stock_price_map(client: RESTClient, symbol: str, date_str: str) -> dict:
    """加载/下载 1s 正股 close，供 Greeks 或 replay 使用。"""
    stock_dir = os.path.join(STOCK_OUTPUT_DIR, symbol)
    os.makedirs(stock_dir, exist_ok=True)
    stock_path = os.path.join(stock_dir, f"{symbol}_{date_str}.parquet")

    stk_df = None
    if os.path.exists(stock_path):
        try:
            stk_df = pd.read_parquet(stock_path)
        except Exception:
            stk_df = None

    if stk_df is None:
        try:
            aggs = list(
                client.list_aggs(
                    ticker=symbol,
                    multiplier=1,
                    timespan="second",
                    from_=date_str,
                    to=date_str,
                    limit=50000,
                )
            )
            if aggs:
                stk_df = pd.DataFrame(
                    [
                        {
                            "ts": a.timestamp / 1000.0,
                            "open": a.open,
                            "high": a.high,
                            "low": a.low,
                            "close": a.close,
                            "volume": a.volume,
                        }
                        for a in aggs
                    ]
                )
                stk_df["timestamp"] = pd.to_datetime(stk_df["ts"], unit="s", utc=True).dt.tz_convert(eastern)
                stk_df.to_parquet(stock_path, index=False)
        except Exception as exc:
            logger.warning("Polygon 1s stock download failed for %s %s: %s", symbol, date_str, exc)

    if stk_df is not None and not stk_df.empty:
        stk_df["ts_int"] = stk_df["ts"].round(0).astype("int64")
        return dict(zip(stk_df["ts_int"], stk_df["close"]))

    month_str = date_str[:7]
    m1_path = (
        f"/home/kingfang007/train_data/spnq_train_resampled/{symbol}/"
        f"regular/09:30-16:00/1min/{month_str}.parquet"
    )
    if not os.path.exists(m1_path):
        return {}

    try:
        m1_df = pd.read_parquet(m1_path)
        m1_df["date_only"] = m1_df["timestamp"].dt.date
        m1_df = m1_df[m1_df["date_only"] == pd.to_datetime(date_str).date()]
        if m1_df.empty:
            return {}
        m1_df = m1_df.set_index("timestamp")
        start_ts = pd.to_datetime(f"{date_str} 09:30:00").tz_localize("America/New_York")
        end_ts = pd.to_datetime(f"{date_str} 16:00:00").tz_localize("America/New_York")
        grid = pd.date_range(start=start_ts, end=end_ts, freq="s", tz="America/New_York")
        m1_1s = m1_df.reindex(grid).ffill().bfill()
        m1_1s["ts"] = m1_1s.index.astype("int64") / 1e9
        return dict(zip(m1_1s["ts"], m1_1s["close"]))
    except Exception as exc:
        logger.debug("1m stock fallback failed for %s %s: %s", symbol, date_str, exc)
        return {}


def download_option_quotes(client: RESTClient, symbol: str, date_str: str, group_df: pd.DataFrame) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for _, row in group_df.iterrows():
        occ = row["contract_symbol"]
        poly_ticker = occ if str(occ).startswith("O:") else f"O:{occ}"
        try:
            quotes = list(
                client.list_quotes(
                    ticker=poly_ticker,
                    timestamp_gte=date_str,
                    timestamp_lte=date_str,
                    limit=50000,
                )
            )
            if not quotes:
                continue

            df = pd.DataFrame(
                [
                    {
                        "timestamp": getattr(q, "sip_timestamp", getattr(q, "participant_timestamp", 0)),
                        "bid": getattr(q, "bid_price", 0.0),
                        "ask": getattr(q, "ask_price", 0.0),
                        "bid_size": getattr(q, "bid_size", 0.0),
                        "ask_size": getattr(q, "ask_size", 0.0),
                    }
                    for q in quotes
                ]
            )
            df = df[(df["bid"] > 0) & (df["ask"] >= df["bid"])].copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ns", utc=True).dt.tz_convert(eastern)

            time_series = df["timestamp"].dt.time
            df = df[(time_series >= datetime.time(9, 30)) & (time_series < datetime.time(16, 0))]
            if df.empty:
                continue

            df["ts_1s"] = df["timestamp"].dt.floor("1s")
            df = df.sort_values("timestamp").drop_duplicates(subset=["ts_1s"], keep="last").copy()
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
            frames.append(df)
        except Exception as exc:
            logger.error("Contract %s process error: %s", occ, exc, exc_info=True)
    return frames


def process_single_day_polygon(args):
    symbol, date_str, group_df, r_val, compute_greeks, download_stock = args
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
        quote_frames = download_option_quotes(client, symbol, date_str, group_df) if DOWNLOAD_OPTIONS else []
        if not quote_frames:
            return f"⚠️ {symbol} {date_str}: No valid quotes."
        final_df = pd.concat(quote_frames, ignore_index=True)

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
    return f"🎯 {symbol} {date_str}: {mode} | {len(final_df)} rows"


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
    parser = argparse.ArgumentParser(description="Polygon 1s option quote sniper")
    parser.add_argument("--target-map", default=TARGET_MAP_FILE)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--compute-greeks", action="store_true", help="optional 1s BSM Greeks (default off)")
    parser.add_argument(
        "--download-stock",
        action="store_true",
        default=None,
        help="download/cache 1s underlying (default on unless --no-download-stock)",
    )
    parser.add_argument("--no-download-stock", action="store_true", help="skip 1s stock download")
    parser.add_argument("--force", action="store_true", help="overwrite existing day files")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global OUTPUT_DIR, FORCE_OVERWRITE
    OUTPUT_DIR = args.output_dir
    FORCE_OVERWRITE = args.force

    compute_greeks = args.compute_greeks or COMPUTE_GREEKS_AT_1S
    if args.no_download_stock:
        download_stock = False
    elif args.download_stock is True:
        download_stock = True
    else:
        download_stock = DOWNLOAD_STOCK_1S

    if not os.path.exists(args.target_map):
        logger.error("Target map not found: %s", args.target_map)
        return

    target_map = pd.read_parquet(args.target_map)
    rfr_series = load_rfr_series() if compute_greeks else None

    symbols = target_map["symbol"].unique()
    logger.info(
        "Polygon 1s sniper | symbols=%d | compute_greeks=%s | download_stock=%s",
        len(symbols),
        compute_greeks,
        download_stock,
    )

    for sym in symbols:
        sym_df = target_map[target_map["symbol"] == sym]
        date_tasks = []
        for d, g in sym_df.groupby("date_str"):
            out_path = os.path.join(OUTPUT_DIR, sym, f"{sym}_{d}.parquet")
            if os.path.exists(out_path) and not FORCE_OVERWRITE:
                continue

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
            date_tasks.append((sym, d, g, r_val, compute_greeks, download_stock))

        if not date_tasks:
            logger.info("Symbol [%s] already fully processed.", sym)
            continue

        logger.info("Processing symbol [%s] for %d days...", sym, len(date_tasks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            list(
                tqdm(
                    executor.map(process_single_day_polygon, date_tasks),
                    total=len(date_tasks),
                    desc=f"Symbol {sym}",
                )
            )

    logger.info("All symbols processed.")


if __name__ == "__main__":
    main()
