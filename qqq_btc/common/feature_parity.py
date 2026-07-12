#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
离线 parquet vs 在线重算 —— 分层特征 parity 审计(秒级,无需 dry_sim)。

replay 用的是 feature_merge 预计算 parquet;实盘/FCS 走另一条路径。
本模块把特征按「预期是否一致」分层,用单元测试/CLI 快速定位偏差来源,
避免每次跑一整天多进程 dry_sim。

分层:
  deterministic  time/trend/open30 — enrich_fcs_bars 应与离线 bit-identical
  price_pandas   OHLC 派生 — FCS _pandas_compute_features 应与离线接近
  vix            vix_level — **已知口径不同**,只检查相关性与文档化差异
  options        期权聚合 — 需完整 chain,单元测试默认 skip
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from qqq_btc.common.time_features import TIME_FEATURE_NAMES
from qqq_btc.common.trend_features import OPEN30_FEATURE_NAMES, TREND_FEATURE_NAMES
from qqq_btc.common.session_history import (
    DEFAULT_CARRYOVER_BARS,  # noqa: F401  (兼容旧引用)
    FEATURE_CARRYOVER_BARS,
    session_tail,
)
from qqq_btc.live.fcs_adapter import enrich_fcs_bars

DETERMINISTIC_FEATURES: tuple[str, ...] = tuple(
    TIME_FEATURE_NAMES + TREND_FEATURE_NAMES + OPEN30_FEATURE_NAMES
)

PRICE_PANDAS_FEATURES: tuple[str, ...] = (
    "close_log_return",
    "vwap_log_return",
    "volume_log",
    "volume_ratio",
    "vwap_diff",
    "garman_klass_vol",
    "return_divergence",
    "bb_width",
    "adx_smooth_10",
)

# POC 离线/在线实现差异大,单独标注不纳入 price 通过率
PRICE_PANDAS_OPTIONAL: tuple[str, ...] = ("poc_deviation",)

VIX_FEATURES: tuple[str, ...] = ("vix_level",)

DEFAULT_OFFLINE_PARQUET = Path(
    "/home/kingfang007/train_data/quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-06.parquet"
)


@dataclass
class FeatureColumnReport:
    feature: str
    tier: str
    rows: int
    med_abs_err: float
    max_abs_err: float
    corr: Optional[float]
    pass_: bool
    note: str = ""


@dataclass
class FeatureParityReport:
    day: str
    rows: int
    columns: List[FeatureColumnReport] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.columns:
            return 0.0
        return sum(1 for c in self.columns if c.pass_) / len(self.columns)

    def to_dict(self) -> dict:
        return {
            "day": self.day,
            "rows": self.rows,
            "pass_rate": self.pass_rate,
            "columns": [
                {
                    "feature": c.feature,
                    "tier": c.tier,
                    "rows": c.rows,
                    "med_abs_err": c.med_abs_err,
                    "max_abs_err": c.max_abs_err,
                    "corr": c.corr,
                    "pass": c.pass_,
                    "note": c.note,
                }
                for c in self.columns
            ],
        }


def _ts_col(df: pd.DataFrame) -> str:
    if "timestamp" in df.columns:
        return "timestamp"
    if "ts" in df.columns:
        return "ts"
    raise ValueError("DataFrame 需含 timestamp 或 ts 列")


def _prepare_day_frame(df: pd.DataFrame, day: str) -> pd.DataFrame:
    ts_col = _ts_col(df)
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], utc=True)
    target = pd.Timestamp(day).date()
    out = out[out[ts_col].dt.date == target].copy()
    if out.empty:
        raise ValueError(f"日期 {day} 在 parquet 中无数据")
    return out.sort_values(ts_col).reset_index(drop=True)


def prepare_fcs_day_frame(
    df: pd.DataFrame,
    day: str,
    *,
    use_carryover: bool = False,
    carryover_bars: int = FEATURE_CARRYOVER_BARS,
) -> pd.DataFrame:
    """
    构造 FCS 对拍用的分钟表。

    use_carryover=False(默认): 仅目标日,模拟 --warmup-from same-day / 无跨日预热。
    use_carryover=True: 在目标日前拼接上一交易日 tail,模拟实盘 history carryover。
    """
    ts_col = _ts_col(df)
    work = df.copy()
    work[ts_col] = pd.to_datetime(work[ts_col], utc=True)
    target = pd.Timestamp(day).date()
    day_df = work[work[ts_col].dt.date == target].copy()
    if day_df.empty:
        raise ValueError(f"日期 {day} 在 parquet 中无数据")
    day_df = day_df.sort_values(ts_col).reset_index(drop=True)
    if not use_carryover:
        return day_df

    prior = work[work[ts_col].dt.date < target].sort_values(ts_col)
    if prior.empty:
        return day_df
    tail = session_tail(prior, carryover_bars)
    if tail.empty:
        return day_df
    return pd.concat([tail, day_df], ignore_index=True).sort_values(ts_col).reset_index(drop=True)


def simulate_fcs_stream_enrich(day_df: pd.DataFrame) -> pd.DataFrame:
    """
    逐 bar 追加 history 并 enrich —— 与 FCS 流式路径一致(因果、无未来函数)。
    返回仅目标日各 bar 的 enrich 结果(与 batch enrich_fcs_bars 末行应对齐)。
    """
    ts_col = _ts_col(day_df)
    target_dates = set(pd.to_datetime(day_df[ts_col]).dt.date)
    hist = pd.DataFrame()
    out_rows: list[pd.Series] = []
    for i in range(len(day_df)):
        row = day_df.iloc[[i]].copy()
        hist = pd.concat([hist, row], ignore_index=True)
        enriched = enrich_fcs_bars(hist, price_col="close")
        last = enriched.iloc[-1]
        if pd.to_datetime(last[ts_col]).date() in target_dates:
            out_rows.append(last)
    if not out_rows:
        return pd.DataFrame()
    return pd.DataFrame(out_rows).reset_index(drop=True)


def _corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if len(a) < 10 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _column_report(
    feature: str,
    tier: str,
    offline: np.ndarray,
    live: np.ndarray,
    *,
    tol: float,
    max_cap: Optional[float] = None,
    expect_divergent: bool = False,
    min_corr: float = 0.5,
    note: str = "",
) -> FeatureColumnReport:
    off = np.nan_to_num(np.asarray(offline, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    liv = np.nan_to_num(np.asarray(live, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.isfinite(off) & np.isfinite(liv)
    n = int(mask.sum())
    if n == 0:
        return FeatureColumnReport(feature, tier, 0, np.nan, np.nan, None, False, note or "no finite rows")

    d = np.abs(off[mask] - liv[mask])
    med = float(np.median(d))
    mx = float(np.max(d))
    corr = _corr(off[mask], liv[mask])

    if expect_divergent:
        ok = corr is None or corr >= min_corr
        if not note:
            note = (
                "已知口径不同:离线=EMA(9/21)+contango on vix_proxy_close; "
                "FCS=(close-mean)/std on VIXY 1m"
            )
        return FeatureColumnReport(feature, tier, n, med, mx, corr, ok, note)

    ok = med <= tol
    if max_cap is not None:
        ok = ok and mx <= max_cap
    return FeatureColumnReport(feature, tier, n, med, mx, corr, ok, note)


def recompute_deterministic_from_ohlc(
    day_df: pd.DataFrame,
    *,
    stream_mode: bool = False,
    target_day: str | None = None,
) -> pd.DataFrame:
    """在线路径:FCS enrich 钩子与 SE bridge 共用 enrich_fcs_bars。"""
    ts_col = _ts_col(day_df)
    base_cols = [c for c in ("open", "high", "low", "close", "volume") if c in day_df.columns]
    if "close" not in base_cols:
        raise ValueError("缺少 close 列,无法 enrich")
    frame = day_df[[ts_col] + base_cols].copy()
    if stream_mode:
        enriched = simulate_fcs_stream_enrich(frame)
    else:
        frame = frame.rename(columns={ts_col: "timestamp"})
        enriched = enrich_fcs_bars(frame)
        enriched = enriched.rename(columns={"timestamp": ts_col})
    if target_day:
        td = pd.Timestamp(target_day).date()
        enriched = enriched[pd.to_datetime(enriched[ts_col]).dt.date == td].copy()
    return enriched.reset_index(drop=True)


def recompute_price_pandas_from_ohlc(day_df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """在线路径:与 FCS 共用 qqq_btc.common.price_features 唯一实现。"""
    from qqq_btc.common.price_features import compute_price_pandas_features

    ts_col = _ts_col(day_df)
    ohlc = day_df.copy()
    ohlc[ts_col] = pd.to_datetime(ohlc[ts_col], utc=True)
    ohlc = ohlc.set_index(ts_col).sort_index()
    out = compute_price_pandas_features(ohlc, list(features))
    return out.reset_index().rename(columns={"index": "timestamp"})


STOCK_MINUTE_VWAP_ROOT = Path.home() / "train_data/spnq_train"


def attach_minute_vwap(frame: pd.DataFrame, symbol: str = "QQQ") -> pd.DataFrame:
    """
    把 vwap 列替换为 resample 源(spnq_train)的交易所分钟 vwap —— 与发球机/实盘
    feed(IBKR bar.wap)同口径。quote_features_raw 存盘的 vwap 是日内累计口径,
    直接喂给 price 特征会让 vwap_log_return 偏差 ~2.5e-3。
    找不到源文件时删除 vwap 列(退回累计口径 fallback)。
    """
    ts_col = _ts_col(frame)
    out = frame.copy()
    ts = pd.to_datetime(out[ts_col], utc=True)
    months = sorted({t.strftime("%Y-%m") for t in ts})
    parts = []
    for month in months:
        p = STOCK_MINUTE_VWAP_ROOT / symbol / f"{month}.parquet"
        if p.exists():
            src = pd.read_parquet(p, columns=["timestamp", "vwap"])
            src["timestamp"] = pd.to_datetime(src["timestamp"], utc=True)
            parts.append(src)
    if not parts:
        return out.drop(columns=["vwap"], errors="ignore")
    ref = pd.concat(parts).drop_duplicates("timestamp").set_index("timestamp")["vwap"]
    out["vwap"] = ts.map(ref)
    return out


def recompute_fcs_vix_level(day_df: pd.DataFrame) -> pd.Series:
    """
    FCS 在线 vix_level:对 vix_proxy_close 做 (last-mean)/std, win=60, >20 bar。
    与离线 generate_vix_level_global 的 rolling(60, min_periods=20) z-score 同公式;
    warm(带前日 VIXY 史)时两者应收敛(实测 corr=1.0),cold 只在开盘前 ~60 根偏差。
    """
    if "vix_proxy_close" not in day_df.columns:
        return pd.Series(0.0, index=np.arange(len(day_df)))
    close = pd.to_numeric(day_df["vix_proxy_close"], errors="coerce").astype(float)
    vals = np.zeros(len(close), dtype=np.float64)
    arr = close.to_numpy()
    for i in range(len(arr)):
        win = arr[max(0, i - 59) : i + 1]
        finite = win[np.isfinite(win)]
        if len(finite) <= 20:
            vals[i] = 0.0
            continue
        sub = finite[-min(len(finite), 60) :]
        vals[i] = (sub[-1] - sub.mean()) / (sub.std() + 1e-6)
    return pd.Series(vals, index=day_df.index)


def compare_feature_tiers(
    offline_day: pd.DataFrame,
    *,
    det_tol: float = 1e-5,
    price_tol: float = 1e-3,
    vix_min_corr: float = 0.5,
    use_carryover: bool = False,
    stream_mode: bool = False,
    target_day: str | None = None,
) -> FeatureParityReport:
    ts_col = _ts_col(offline_day)
    day = target_day or str(pd.to_datetime(offline_day[ts_col].iloc[0]).date())
    offline_target = offline_day
    if target_day:
        offline_target = offline_day[
            pd.to_datetime(offline_day[ts_col]).dt.date == pd.Timestamp(target_day).date()
        ].copy().reset_index(drop=True)

    work_frame = offline_day if use_carryover else offline_target
    enriched = recompute_deterministic_from_ohlc(
        work_frame,
        stream_mode=stream_mode,
        target_day=day,
    )
    price_live = recompute_price_pandas_from_ohlc(
        attach_minute_vwap(work_frame), PRICE_PANDAS_FEATURES
    )
    if "timestamp" in price_live.columns:
        price_live = price_live[
            pd.to_datetime(price_live["timestamp"]).dt.date == pd.Timestamp(day).date()
        ].copy().reset_index(drop=True)
    # vix 与 price 同口径:warm 时带前史,再切回目标日
    vix_live_full = recompute_fcs_vix_level(work_frame.reset_index(drop=True))
    ts_all = pd.to_datetime(work_frame.reset_index(drop=True)[ts_col])
    vix_live = vix_live_full[ts_all.dt.date == pd.Timestamp(day).date()].reset_index(drop=True)

    reports: List[FeatureColumnReport] = []

    for feat in DETERMINISTIC_FEATURES:
        if feat not in offline_target.columns or feat not in enriched.columns:
            continue
        reports.append(
            _column_report(
                feat,
                "deterministic",
                pd.to_numeric(offline_target[feat], errors="coerce").to_numpy(),
                pd.to_numeric(enriched[feat], errors="coerce").to_numpy(),
                tol=det_tol,
                max_cap=0.05,
            )
        )

    for feat in PRICE_PANDAS_FEATURES + PRICE_PANDAS_OPTIONAL:
        if feat not in offline_target.columns or feat not in price_live.columns:
            continue
        tier = "price_pandas_optional" if feat in PRICE_PANDAS_OPTIONAL else "price_pandas"
        reports.append(
            _column_report(
                feat,
                tier,
                pd.to_numeric(offline_target[feat], errors="coerce").to_numpy(),
                pd.to_numeric(price_live[feat], errors="coerce").to_numpy(),
                tol=price_tol,
                max_cap=1.0 if feat in PRICE_PANDAS_OPTIONAL else 0.05,
                note="POC 离线/在线实现不同" if feat in PRICE_PANDAS_OPTIONAL else "",
                expect_divergent=feat in PRICE_PANDAS_OPTIONAL,
                min_corr=0.0 if feat in PRICE_PANDAS_OPTIONAL else 0.5,
            )
        )

    if "vix_level" in offline_target.columns:
        reports.append(
            _column_report(
                "vix_level",
                "vix_known_divergent",
                pd.to_numeric(offline_target["vix_level"], errors="coerce").to_numpy(),
                vix_live.to_numpy(),
                tol=0.0,
                expect_divergent=True,
                min_corr=vix_min_corr,
            )
        )

    return FeatureParityReport(day=day, rows=len(offline_target), columns=reports)


def audit_offline_parquet_day(
    parquet_path: Path | str,
    day: str,
    *,
    det_tol: float = 1e-5,
    price_tol: float = 1e-3,
    vix_min_corr: float = 0.5,
    use_carryover: bool = False,
    stream_mode: bool = False,
) -> FeatureParityReport:
    df = pd.read_parquet(parquet_path)
    day_df = prepare_fcs_day_frame(df, day, use_carryover=use_carryover)
    return compare_feature_tiers(
        day_df,
        det_tol=det_tol,
        price_tol=price_tol,
        vix_min_corr=vix_min_corr,
        use_carryover=use_carryover,
        stream_mode=stream_mode,
        target_day=day,
    )


def audit_fcs_parity_day(
    parquet_path: Path | str,
    day: str,
    *,
    det_tol: float = 1e-5,
    price_tol: float = 1e-3,
    vix_min_corr: float = 0.5,
) -> dict:
    """
    无预热 vs 有预热 FCS 特征对拍摘要(秒级,无需 dry_sim)。

    返回 cold(no_carryover) / warm(carryover) / stream(cold+逐bar) 三份报告。
    """
    cold = audit_offline_parquet_day(
        parquet_path, day,
        det_tol=det_tol, price_tol=price_tol, vix_min_corr=vix_min_corr,
        use_carryover=False, stream_mode=False,
    )
    warm = audit_offline_parquet_day(
        parquet_path, day,
        det_tol=det_tol, price_tol=price_tol, vix_min_corr=vix_min_corr,
        use_carryover=True, stream_mode=False,
    )
    stream = audit_offline_parquet_day(
        parquet_path, day,
        det_tol=det_tol, price_tol=price_tol, vix_min_corr=vix_min_corr,
        use_carryover=False, stream_mode=True,
    )
    return {
        "day": day,
        "cold_no_carryover": cold.to_dict(),
        "warm_with_carryover": warm.to_dict(),
        "stream_incremental": stream.to_dict(),
    }


def format_report_summary(report: FeatureParityReport) -> str:
    lines = [f"day={report.day} rows={report.rows} pass_rate={report.pass_rate:.1%}", ""]
    for tier in ("deterministic", "price_pandas", "price_pandas_optional", "vix_known_divergent"):
        cols = [c for c in report.columns if c.tier == tier]
        if not cols:
            continue
        lines.append(f"[{tier}]")
        for c in cols:
            status = "PASS" if c.pass_ else "FAIL"
            corr = f" corr={c.corr:.3f}" if c.corr is not None else ""
            lines.append(
                f"  {status} {c.feature:28s} med_err={c.med_abs_err:.6g} max_err={c.max_abs_err:.6g}{corr}"
            )
            if c.note:
                lines.append(f"         {c.note}")
        lines.append("")
    return "\n".join(lines).rstrip()
