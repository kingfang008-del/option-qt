#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 Massive/Polygon 查 QQQ 当日开盘价，锁 4 个 1DTE 合约（对齐 4-bucket 语义）。

不再依赖「先下全市场交易/day_iv → 再提合约」；直接：

  1) 取 underlying 当日 open（优先日线 open，缺则 09:30 分钟 open）
  2) 解析 trading-DTE==prefer 的到期（默认 1）
  3) 在链上按 ATM(|Δ|≈0.50) / OTM(|Δ|≈0.25) 目标行权价就近选 Put/Call

4-bucket（与 anchor_qqq_1dte_4bucket / old lock 一致）:
  0 Put ATM(0.50) | 1 Put OTM(0.25) | 2 Call ATM(0.50) | 3 Call OTM(0.25)

用法:
  export MASSIVE_API_KEY=...   # 或 POLYGON_API_KEY
  python preprocess/download/step1_lock_4bucket_from_open.py \\
      --start-date 2026-07-14 --end-date 2026-07-14 \\
      --config preprocess/CONFIG/anchor_qqq_1dte_4bucket.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from polygon import RESTClient
from scipy.stats import norm
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PREPROCESS_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _PREPROCESS_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from preprocess.download.build_0dte_api_ladder_map import get_contract_rows  # noqa: E402
from preprocess.download.build_mag7_short_dte_api_ladder_map import (  # noqa: E402
    resolve_expiry_for_dte,
    trading_dates_union,
)
from preprocess.download.step1_build_target_map_old import (  # noqa: E402
    bucket_targets,
    load_anchor_config,
    select_front_dte,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("lock_4bucket_from_open")

NY = "America/New_York"
DEFAULT_CONFIG = _PREPROCESS_ROOT / "CONFIG" / "anchor_qqq_1dte_4bucket.json"


def resolve_api_key(explicit: str | None = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    for k in ("MASSIVE_API_KEY", "POLYGON_API_KEY", "POLYGON_KEY"):
        v = os.environ.get(k, "").strip()
        if v:
            return v
    raise SystemExit("缺少 API key：请设置 MASSIVE_API_KEY 或 POLYGON_API_KEY")


def stock_open_price(client: RESTClient, symbol: str, date_str: str) -> Optional[float]:
    """优先日线 open；失败再退到 RTH 第一根 1m open。"""
    try:
        bars = list(
            client.list_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=date_str,
                to=date_str,
                limit=5,
                adjusted=True,
            )
        )
        if bars:
            return float(bars[0].open)
    except Exception as exc:
        logger.warning("day open failed %s %s: %s", symbol, date_str, exc)

    try:
        bars = list(
            client.list_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="minute",
                from_=date_str,
                to=date_str,
                limit=50000,
                adjusted=True,
            )
        )
    except Exception as exc:
        logger.warning("minute open fallback failed %s %s: %s", symbol, date_str, exc)
        return None
    rows = []
    for b in bars:
        ts = pd.Timestamp(b.timestamp, unit="ms", tz="UTC").tz_convert(NY)
        if ts.strftime("%H:%M") >= "09:30":
            rows.append({"ts": ts, "open": float(b.open)})
    if not rows:
        return None
    return float(pd.DataFrame(rows).sort_values("ts").iloc[0]["open"])


def target_strike_from_delta(
    spot: float,
    *,
    is_call: bool,
    target_abs_delta: float,
    iv: float,
    t_years: float,
    r: float = 0.045,
) -> float:
    """用 BSM 近似把 |Δ| 目标映射到行权价（无链上希腊时的开盘价锁约）。"""
    if spot <= 0:
        return spot
    if abs(float(target_abs_delta) - 0.50) < 1e-6 or t_years <= 0 or iv <= 0:
        return float(spot)
    z = float(norm.ppf(float(target_abs_delta)))
    # call Δ=N(d1)；put |Δ|=N(-d1) → d1 = -N^{-1}(|Δ|)
    d1 = z if is_call else -z
    s = float(iv)
    ln_sk = d1 * s * math.sqrt(t_years) - (r + 0.5 * s * s) * t_years
    return float(spot * math.exp(-ln_sk))


def pick_closest_contract(
    chain: pd.DataFrame,
    *,
    is_call: bool,
    target_k: float,
    prefer_otm: bool,
    spot: float,
) -> Optional[pd.Series]:
    typ = "call" if is_call else "put"
    sub = chain[chain["contract_type"].astype(str).str.lower() == typ].copy()
    if sub.empty:
        return None
    if prefer_otm:
        if is_call:
            otm = sub[sub["strike_price"] >= spot]
        else:
            otm = sub[sub["strike_price"] <= spot]
        if not otm.empty:
            sub = otm
    sub = sub.copy()
    sub["k_dist"] = (sub["strike_price"].astype(float) - float(target_k)).abs()
    return sub.sort_values(["k_dist", "strike_price"]).iloc[0]


def lock_day_4bucket(
    client: RESTClient,
    symbol: str,
    date_str: str,
    cfg: dict[str, Any],
    *,
    expiry_cache: dict[str, bool],
    assume_iv: float = 0.22,
    rfr: float = 0.045,
) -> list[dict[str, Any]]:
    spot = stock_open_price(client, symbol, date_str)
    if spot is None or spot <= 0:
        logger.warning("%s %s: no open price", symbol, date_str)
        return []

    prefer = int(cfg.get("front_prefer_dte", 1))
    allowed = [int(x) for x in (cfg.get("front_allowed_dte") or [prefer])]
    # 尝试 prefer，再在 allowed 内退档（与 old select_front_dte 语义一致）
    available: list[int] = []
    exp_by_dte: dict[int, str] = {}
    for dte in sorted(set(allowed + [prefer, 0, 1, 2])):
        exp = resolve_expiry_for_dte(client, symbol, date_str, dte, expiry_cache)
        if exp:
            available.append(dte)
            exp_by_dte[dte] = exp
    front_dte = select_front_dte(available, cfg)
    if front_dte is None:
        logger.warning("%s %s: no front DTE in allowed=%s", symbol, date_str, allowed)
        return []
    expiration = exp_by_dte[int(front_dte)]
    chain = get_contract_rows(client, symbol, expiration)
    if chain.empty:
        logger.warning("%s %s: empty chain exp=%s", symbol, date_str, expiration)
        return []

    t_years = max(float(front_dte) / 252.0, 1.0 / 252.0)
    rows: list[dict[str, Any]] = []
    for b_id, _is_front, is_call, target_delta in bucket_targets(cfg):
        target_k = target_strike_from_delta(
            spot,
            is_call=bool(is_call),
            target_abs_delta=float(target_delta),
            iv=float(assume_iv),
            t_years=t_years,
            r=float(rfr),
        )
        prefer_otm = abs(float(target_delta) - 0.50) > 1e-6
        picked = pick_closest_contract(
            chain,
            is_call=bool(is_call),
            target_k=target_k,
            prefer_otm=prefer_otm,
            spot=spot,
        )
        if picked is None:
            logger.warning(
                "%s %s: bucket %s missing side=%s",
                symbol,
                date_str,
                b_id,
                "CALL" if is_call else "PUT",
            )
            return []
        strike = float(picked["strike_price"])
        ticker = str(picked["ticker"])
        side = "CALL" if is_call else "PUT"
        tag = f"{side}_{'ATM' if abs(float(target_delta) - 0.50) < 1e-6 else 'OTM'}"
        rows.append(
            {
                "date_str": date_str,
                "contract_symbol": ticker if ticker.startswith("O:") else f"O:{ticker.replace('O:', '')}",
                "bucket_id": int(b_id),
                "symbol": symbol,
                "tag": tag,
                "side": side,
                "target_abs_delta": float(target_delta),
                "target_dte": int(front_dte),
                "selected_dte": int(front_dte),
                "front_dte": int(front_dte),
                "expiration": expiration,
                "strike": strike,
                "stock_open": float(spot),
                "stock_close_at_lock": float(spot),
                "target_strike": float(target_k),
                "assume_iv": float(assume_iv),
                "moneyness_at_lock": math.log(strike / spot) if strike > 0 else float("nan"),
                "lock_timestamp": pd.Timestamp(f"{date_str} 09:30:00", tz=NY).isoformat(),
                "lock_mode": "open_price_4bucket",
            }
        )

    if bool(cfg.get("require_complete_buckets", True)) and len(rows) < len(bucket_targets(cfg)):
        return []
    # step2 兼容：contract_symbol 可带或不带 O:
    for r in rows:
        r["contract_symbol"] = str(r["contract_symbol"]).replace("O:", "")
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--symbols", default="QQQ")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument(
        "--output",
        default=str(Path.home() / "train_data/locked_targets_map_open_4bucket.parquet"),
    )
    p.add_argument("--assume-iv", type=float, default=0.22)
    p.add_argument("--rfr", type=float, default=0.045)
    p.add_argument("--api-key", default="")
    p.add_argument("--report", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_absolute() and not cfg_path.exists():
        alt = _PREPROCESS_ROOT / "CONFIG" / cfg_path.name
        cfg_path = alt if alt.exists() else cfg_path
    cfg = load_anchor_config(cfg_path)
    api_key = resolve_api_key(args.api_key)
    client = RESTClient(api_key)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    all_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for symbol in symbols:
        dates = trading_dates_union(symbol, args.start_date, args.end_date, client=client)
        logger.info(
            "%s days=%d (%s..%s) profile=%s prefer_dte=%s",
            symbol,
            len(dates),
            dates[0] if dates else None,
            dates[-1] if dates else None,
            cfg.get("profile"),
            cfg.get("front_prefer_dte"),
        )
        expiry_cache: dict[str, bool] = {}
        for d in tqdm(dates, desc=f"{symbol}-open4b"):
            rows = lock_day_4bucket(
                client,
                symbol,
                d,
                cfg,
                expiry_cache=expiry_cache,
                assume_iv=float(args.assume_iv),
                rfr=float(args.rfr),
            )
            if rows:
                all_rows.extend(rows)
            else:
                missing.append({"symbol": symbol, "date_str": d})

    if not all_rows:
        raise SystemExit("no contracts locked")

    out = (
        pd.DataFrame(all_rows)
        .sort_values(["symbol", "date_str", "bucket_id"])
        .reset_index(drop=True)
    )
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False, compression="zstd")
    report = {
        "config": str(cfg_path),
        "profile": cfg.get("profile"),
        "output": str(out_path),
        "n_rows": int(len(out)),
        "n_days": int(out["date_str"].nunique()),
        "missing_days": missing,
        "sample": out.head(8).to_dict(orient="records"),
    }
    rep_path = Path(args.report).expanduser() if args.report else out_path.with_suffix(".report.json")
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("wrote %s rows=%d days=%d", out_path, len(out), out["date_str"].nunique())
    logger.info("report → %s", rep_path)
    if "front_dte" in out.columns:
        logger.info(
            "front_dte dist:\n%s",
            out.groupby("date_str")["front_dte"].first().value_counts().sort_index().to_string(),
        )


if __name__ == "__main__":
    main()
