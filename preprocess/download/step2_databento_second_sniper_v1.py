"""
Databento OPRA 秒级期权 Quote 下载器

读取 locked_targets_map_0dte.parquet 中的目标合约，按 (symbol, date_str) 批量拉取
OPRA.PILLAR quote，输出与 step2_polygon_second_sniper_v1 兼容的 1 秒 parquet。
每行含 source_schema 列（cbbo-1s / cmbp-1），便于后续分析数据质量。

硬约束 MIN_DATE=2023-03-28：不下载 cbbo-1m 分钟级区间，仅 cmbp-1 / cbbo-1s。

Schema 按日期自动选择（通过 metadata.get_dataset_range 探测）:
  - cbbo-1s  : 原生 1 秒 consolidated BBO（2025-02-20 起，视账号而定）
  - cmbp-1   : tick 级 quote，重采样到 1 秒（2023-03-28 起）

用法:
  export DATABENTO_API_KEY=db-xxxxxxxx
  python step2_databento_second_sniper_v1.py
  python step2_databento_second_sniper_v1.py --limit-days 3 --dry-run
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import logging
import os
import re
import time
from typing import Optional

import numpy as np
import pandas as pd
from pytz import timezone

try:
    import databento as db
except ImportError as exc:
    raise SystemExit(
        "缺少 databento 包，请先安装: pip install databento"
    ) from exc

# ================= 默认配置 =================
DEFAULT_TARGET_MAP = os.path.expanduser(
    "~/train_data/locked_targets_map_0dte.parquet"
)
DEFAULT_OUTPUT_DIR = "/mnt/s990/data/raw_1s/options_databento"
DATASET = "OPRA.PILLAR"
# 硬约束：不下载此日期之前的 target（仅有 cbbo-1m 分钟级）
MIN_DATE = "2023-03-28"
# 优先级：秒级原生 > tick 重采样（不含 cbbo-1m）
SCHEMA_PRIORITY = ("cbbo-1s", "cmbp-1")
RTH_START = datetime.time(9, 30)
RTH_END = datetime.time(16, 0)
MAX_RETRIES = 3
RETRY_SLEEP_SEC = 5.0
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger("databento").setLevel(logging.WARNING)
logger = logging.getLogger("Databento_1s_Sniper")
EASTERN = timezone("America/New_York")

OCC_RE = re.compile(r"^O:([A-Z]+)(\d{6}[CP]\d{8})$")
OCC_BARE_RE = re.compile(r"^([A-Z]+)(\d{6}[CP]\d{8})$")
# Databento OPRA raw_symbol: 21-char OSI, root right-padded to 6 chars
OSI_SUFFIX_RE = re.compile(r"^(\d{6}[CP]\d{8})$")


def normalize_occ_symbol(raw: str) -> str:
    s = str(raw).strip()
    if s.startswith("O:"):
        return s
    return f"O:{s}"


def occ_to_databento_symbol(occ: str) -> str:
    """O:QQQ260206P00598000 -> QQQ   260206P00598000 (21-char OSI)"""
    s = normalize_occ_symbol(occ).replace("O:", "")
    m = OCC_BARE_RE.match(s)
    if not m:
        raise ValueError(f"无法解析 OCC 合约: {occ}")
    root = m.group(1)
    suffix = m.group(2)
    return f"{root:<6}{suffix}"


def databento_symbol_to_occ(db_symbol: str) -> str:
    """QQQ   260206P00598000 -> O:QQQ260206P00598000"""
    s = str(db_symbol).strip()
    if s.startswith("O:"):
        return s

    compact = s.replace(" ", "")
    m = OCC_BARE_RE.match(compact)
    if m:
        return f"O:{m.group(1)}{m.group(2)}"

    if len(s) == 21:
        root = s[:6].rstrip()
        suffix = s[6:]
        if OSI_SUFFIX_RE.match(suffix):
            return f"O:{root}{suffix}"

    m2 = re.match(r"^([A-Z]+)\s+(\d{6}[CP]\d{8})$", s)
    if m2:
        return f"O:{m2.group(1)}{m2.group(2)}"

    return f"O:{compact}" if not s.startswith("O:") else s


def parse_strike_from_occ(occ: str) -> float:
    m = re.search(r"[CP](\d{8})$", normalize_occ_symbol(occ).replace("O:", ""))
    return float(m.group(1)) / 1000.0 if m else 0.0


def load_schema_ranges(client: db.Historical) -> dict[str, dict[str, pd.Timestamp]]:
    """从 Databento metadata 读取各 schema 可用日期区间。"""
    meta = client.metadata.get_dataset_range(dataset=DATASET)
    raw = meta.get("schema") or {}
    ranges: dict[str, dict[str, pd.Timestamp]] = {}
    for schema, bounds in raw.items():
        if not isinstance(bounds, dict):
            continue
        start = bounds.get("start")
        end = bounds.get("end")
        if start is None or end is None:
            continue
        ranges[schema] = {
            "start": pd.Timestamp(start),
            "end": pd.Timestamp(end),
        }
    return ranges


def _to_date(ts: pd.Timestamp | str) -> datetime.date:
    """统一转为 date 比较，避免 tz-naive / tz-aware 混比。"""
    t = pd.Timestamp(ts)
    if t.tz is not None:
        t = t.tz_convert("UTC")
    return t.date()


def pick_schema_for_date(
    date_str: str,
    schema_ranges: dict[str, dict[str, pd.Timestamp]],
) -> str:
    day = _to_date(date_str)
    if day < _to_date(MIN_DATE):
        raise ValueError(
            f"{date_str} 早于 MIN_DATE={MIN_DATE}，拒绝使用 cbbo-1m 分钟级数据"
        )
    for schema in SCHEMA_PRIORITY:
        bounds = schema_ranges.get(schema)
        if bounds is None:
            continue
        if _to_date(bounds["start"]) <= day < _to_date(bounds["end"]):
            return schema
    available = {
        s: f"{b['start'].date()} ~ {b['end'].date()}"
        for s, b in schema_ranges.items()
        if s in SCHEMA_PRIORITY
    }
    raise ValueError(
        f"{date_str} 无可用 quote schema。"
        f" 当前账号 OPRA 覆盖: {available}"
    )


def _price_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    s = pd.to_numeric(df[col], errors="coerce").astype(float)
    if s.abs().max(skipna=True) > 1e6:
        s = s / 1e9
    return s.fillna(0.0)


def _size_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)


def _build_base_quote_df(
    raw_df: pd.DataFrame,
    symbol: str,
    date_str: str,
    occ_to_meta: dict[str, dict],
) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    ts_col = None
    for candidate in ("ts_event", "ts_recv", "timestamp"):
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        logger.warning("%s %s: 找不到时间戳列", symbol, date_str)
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.tz_convert(EASTERN)

    t = df["timestamp"].dt.time
    df = df[(t >= RTH_START) & (t < RTH_END)].copy()
    if df.empty:
        return pd.DataFrame()

    df["bid"] = _price_series(df, "bid_px_00")
    df["ask"] = _price_series(df, "ask_px_00")
    df["bid_size"] = _size_series(df, "bid_sz_00")
    df["ask_size"] = _size_series(df, "ask_sz_00")

    df = df[(df["bid"] > 0) | (df["ask"] > 0)].copy()
    if df.empty:
        return pd.DataFrame()

    df["mid_price"] = np.where(
        (df["bid"] > 0) & (df["ask"] > 0),
        (df["bid"] + df["ask"]) / 2.0,
        df[["bid", "ask"]].max(axis=1),
    )
    df["price"] = df["mid_price"]

    if "symbol" not in df.columns:
        logger.warning("%s %s: 响应缺少 symbol 列", symbol, date_str)
        return pd.DataFrame()

    df["occ_symbol"] = df["symbol"].map(databento_symbol_to_occ)
    df["ticker"] = df["occ_symbol"].str.replace("O:", "", regex=False)
    df["underlying"] = symbol
    df["bucket_id"] = df["occ_symbol"].map(
        lambda x: occ_to_meta.get(x, {}).get("bucket_id", -1)
    )
    df["tag"] = df["occ_symbol"].map(
        lambda x: occ_to_meta.get(x, {}).get("tag", "")
    )
    df["strike"] = df["occ_symbol"].map(parse_strike_from_occ)
    return df


def _resample_to_1s(
    df: pd.DataFrame,
    date_str: str,
    source_schema: str,
) -> pd.DataFrame:
    if df.empty:
        return df

    if source_schema == "cbbo-1m":
        start_ts = pd.Timestamp(f"{date_str} 09:30:00", tz=EASTERN)
        end_ts = pd.Timestamp(f"{date_str} 15:59:59", tz=EASTERN)
        grid = pd.date_range(start=start_ts, end=end_ts, freq="s")
        parts = []
        value_cols = [
            c
            for c in (
                "bid",
                "ask",
                "bid_size",
                "ask_size",
                "mid_price",
                "price",
                "ticker",
                "tag",
                "bucket_id",
                "underlying",
                "strike",
                "occ_symbol",
            )
            if c in df.columns
        ]
        for _, sub in df.groupby("occ_symbol", sort=False):
            sub = sub.sort_values("timestamp").drop_duplicates(
                subset=["timestamp"], keep="last"
            )
            sub = sub.set_index("timestamp")
            # 仅 ffill：每秒钟只用「已发生」的最近一条分钟 quote，因果安全。
            # 不用 bfill：开盘前若干秒会用未来首条分钟 bar 回填，造成 look-ahead。
            sub_1s = sub.reindex(grid)[value_cols].ffill()
            sub_1s = sub_1s.dropna(subset=["bid", "ask"], how="all")
            if sub_1s.empty:
                continue
            sub_1s = sub_1s.reset_index(names="timestamp")
            parts.append(sub_1s)
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True)
    else:
        df["ts_1s"] = df["timestamp"].dt.floor("1s")
        df = df.sort_values("timestamp").drop_duplicates(
            subset=["occ_symbol", "ts_1s"], keep="last"
        )
        df["timestamp"] = df["ts_1s"]
        df = df.drop(columns=["ts_1s"])

    df["ts"] = df["timestamp"].astype("int64") / 1e9
    df["source_schema"] = source_schema
    out_cols = [
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
        "source_schema",
    ]
    return df[[c for c in out_cols if c in df.columns]]


def transform_quote_df(
    raw_df: pd.DataFrame,
    symbol: str,
    date_str: str,
    occ_to_meta: dict[str, dict],
    source_schema: str,
) -> pd.DataFrame:
    if source_schema == "cbbo-1m":
        raise ValueError(
            f"{date_str}: cbbo-1m 已被 MIN_DATE={MIN_DATE} 硬约束禁止"
        )
    df = _build_base_quote_df(raw_df, symbol, date_str, occ_to_meta)
    return _resample_to_1s(df, date_str, source_schema)


def fetch_day_quotes(
    client: db.Historical,
    db_symbols: list[str],
    date_str: str,
    schema: str,
) -> pd.DataFrame:
    start = f"{date_str}T09:30-04:00"
    end = f"{date_str}T16:00-04:00"

    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = client.timeseries.get_range(
                dataset=DATASET,
                schema=schema,
                stype_in="raw_symbol",
                symbols=db_symbols,
                start=start,
                end=end,
            )
            return data.to_df()
        except Exception as exc:
            last_err = exc
            logger.warning(
                "Databento 请求失败 (%s, schema=%s, attempt %d/%d): %s",
                date_str,
                schema,
                attempt,
                MAX_RETRIES,
                exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SEC * attempt)
    raise RuntimeError(
        f"Databento 下载失败 {date_str} [{schema}]: {last_err}"
    ) from last_err


def process_single_day(args: tuple) -> str:
    symbol, date_str, group_df, api_key, output_dir, force, schema_ranges = args

    out_dir = os.path.join(output_dir, symbol)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{symbol}_{date_str}.parquet")

    if os.path.exists(out_path) and not force:
        return f"⏩ {symbol} {date_str} exists"

    occ_symbols = [normalize_occ_symbol(x) for x in group_df["contract_symbol"]]
    occ_to_meta = {
        occ: {
            "bucket_id": int(row["bucket_id"]),
            "tag": row.get("tag", "") if isinstance(row.get("tag", ""), str) else "",
        }
        for occ, (_, row) in zip(occ_symbols, group_df.iterrows())
    }

    try:
        db_symbols = [occ_to_databento_symbol(x) for x in occ_symbols]
        schema = pick_schema_for_date(date_str, schema_ranges)
    except ValueError as exc:
        return f"❌ {symbol} {date_str}: {exc}"

    client = db.Historical(api_key)
    try:
        raw_df = fetch_day_quotes(client, db_symbols, date_str, schema)
    except Exception as exc:
        return f"❌ {symbol} {date_str}: {exc}"

    final_df = transform_quote_df(
        raw_df, symbol, date_str, occ_to_meta, schema
    )
    if final_df.empty:
        return f"⚠️ {symbol} {date_str}: No valid quotes."

    final_df.to_parquet(
        out_path, engine="pyarrow", index=False, compression="zstd"
    )
    n_contracts = final_df["ticker"].nunique()
    return (
        f"🎯 {symbol} {date_str} [{schema}]: {len(final_df)} rows, "
        f"{n_contracts}/{len(occ_symbols)} contracts"
    )


def build_tasks(
    target_map: pd.DataFrame,
    output_dir: str,
    force: bool,
    symbol_filter: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
    limit_days: Optional[int],
    min_date: str = MIN_DATE,
) -> list[tuple]:
    df = target_map.copy()
    if symbol_filter:
        df = df[df["symbol"] == symbol_filter.upper()]

    effective_from = min_date
    if date_from:
        effective_from = max(date_from, min_date)
    df = df[df["date_str"] >= effective_from]

    if date_to:
        df = df[df["date_str"] <= date_to]

    tasks = []
    seen_days = 0
    for sym in sorted(df["symbol"].unique()):
        sym_df = df[df["symbol"] == sym]
        for d, g in sym_df.groupby("date_str", sort=True):
            if limit_days is not None and seen_days >= limit_days:
                break
            out_path = os.path.join(output_dir, sym, f"{sym}_{d}.parquet")
            if os.path.exists(out_path) and not force:
                continue
            tasks.append((sym, d, g))
            seen_days += 1
        if limit_days is not None and seen_days >= limit_days:
            break
    return tasks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Databento OPRA 1s quote sniper")
    p.add_argument(
        "--target-map",
        default=DEFAULT_TARGET_MAP,
        help="目标合约 parquet 路径",
    )
    p.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("DATABENTO_API_KEY"),
        help="Databento API Key（默认读 DATABENTO_API_KEY 环境变量）",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="并发下载线程数",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载已存在文件",
    )
    p.add_argument(
        "--symbol",
        default=None,
        help="只处理指定 underlying，例如 QQQ",
    )
    p.add_argument("--date-from", default=None, help="起始日期 YYYY-MM-DD")
    p.add_argument("--date-to", default=None, help="结束日期 YYYY-MM-DD")
    p.add_argument(
        "--limit-days",
        type=int,
        default=None,
        help="最多处理 N 个交易日（调试用）",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印任务，不实际下载",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.target_map):
        logger.error("目标文件不存在: %s", args.target_map)
        return

    target_map = pd.read_parquet(args.target_map)
    required = {"date_str", "contract_symbol", "bucket_id", "symbol"}
    missing = required - set(target_map.columns)
    if missing:
        logger.error("目标 parquet 缺少列: %s", sorted(missing))
        return

    all_days = target_map["date_str"].nunique()
    eligible_days = target_map[target_map["date_str"] >= MIN_DATE]["date_str"].nunique()
    skipped_days = all_days - eligible_days
    if skipped_days > 0:
        logger.info(
            "硬约束 MIN_DATE=%s: 跳过 %d / %d 个交易日（分钟级区间）",
            MIN_DATE,
            skipped_days,
            all_days,
        )
    if args.date_from and args.date_from < MIN_DATE:
        logger.warning(
            "--date-from %s 早于 MIN_DATE，实际从 %s 开始",
            args.date_from,
            MIN_DATE,
        )

    tasks = build_tasks(
        target_map,
        args.output_dir,
        args.force,
        args.symbol,
        args.date_from,
        args.date_to,
        args.limit_days,
    )

    logger.info(
        "目标文件: %s | 总行数: %d | 待下载天数: %d | 输出: %s",
        args.target_map,
        len(target_map),
        len(tasks),
        args.output_dir,
    )

    if args.dry_run:
        schema_ranges = {}
        if args.api_key:
            try:
                schema_ranges = load_schema_ranges(db.Historical(args.api_key))
                for schema in SCHEMA_PRIORITY:
                    bounds = schema_ranges.get(schema)
                    if bounds:
                        logger.info(
                            "Schema %s: %s ~ %s",
                            schema,
                            bounds["start"].date(),
                            bounds["end"].date(),
                        )
            except Exception as exc:
                logger.warning("无法读取 schema 覆盖范围: %s", exc)
        for sym, d, g in tasks[:20]:
            contracts = [
                f"{c} -> {occ_to_databento_symbol(c)}"
                for c in g["contract_symbol"]
            ]
            schema_hint = ""
            if schema_ranges:
                try:
                    schema_hint = f" schema={pick_schema_for_date(d, schema_ranges)}"
                except ValueError:
                    schema_hint = " schema=NONE"
            logger.info("[dry-run] %s %s%s -> %s", sym, d, schema_hint, contracts)
        if len(tasks) > 20:
            logger.info("... 还有 %d 天", len(tasks) - 20)
        return

    if not args.api_key:
        logger.error(
            "未设置 Databento API Key。请 export DATABENTO_API_KEY=db-xxx "
            "或使用 --api-key"
        )
        return

    if not tasks:
        logger.info("没有待处理任务，已全部完成。")
        return

    client = db.Historical(args.api_key)
    schema_ranges = load_schema_ranges(client)
    for schema in SCHEMA_PRIORITY:
        bounds = schema_ranges.get(schema)
        if bounds:
            logger.info(
                "Schema %s 可用: %s ~ %s",
                schema,
                bounds["start"].date(),
                bounds["end"].date(),
            )

    payload = [
        (sym, d, g, args.api_key, args.output_dir, args.force, schema_ranges)
        for sym, d, g in tasks
    ]

    ok, skip, warn, err = 0, 0, 0, 0
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.max_workers)
    ) as executor:
        from tqdm import tqdm

        for msg in tqdm(
            executor.map(process_single_day, payload),
            total=len(payload),
            desc="Databento 1s",
        ):
            if msg.startswith("🎯"):
                ok += 1
            elif msg.startswith("⏩"):
                skip += 1
            elif msg.startswith("⚠️"):
                warn += 1
                logger.warning(msg)
            else:
                err += 1
                logger.error(msg)

    logger.info(
        "完成: success=%d skip=%d warn=%d error=%d",
        ok,
        skip,
        warn,
        err,
    )


if __name__ == "__main__":
    main()
