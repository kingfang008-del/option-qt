#!/usr/bin/env python3
"""周状态相似检索 —— 判断「当前一周」更像历史上哪段，而不是盲目用近 2 月微调。

仅用现货分钟行情（QQQ + VIXY），不含期权收益/标签，避免泄漏。
用途：
  1) 诊断：本周 vs 近 N 月中心距离，是否像 regime 突变
  2) 检索：Top-K 相似历史周（及对应日历月），供微调窗混合/压力测试参考
  3) 不自动改权重；输出建议供人工或 weekly_finetune 选用

默认数据:
  ~/train_data/spnq_train_resampled/{QQQ,VIXY}/regular/09:30-16:00/1min/*.parquet

用法:
  python qqq_btc/tools/match_week_regime.py --query 2026-07-01:2026-07-08
  python qqq_btc/tools/match_week_regime.py --query 2026-07-01:2026-07-10 --top 20 \\
      --out qqq_btc/results/regime_match_jul_w1
  python qqq_btc/tools/match_week_regime.py --query 2026-07-01:2026-07-08 --recent-months 2
  python qqq_btc/tools/match_week_regime.py --query 2026-07-01:2026-07-08 --emit-train-months --quiet

下游:
  weekly_finetune.py --suggest-train-months --regime-query 2026-07-01:2026-07-08
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_SPOT_ROOT = Path.home() / "train_data" / "spnq_train_resampled"

# 周向量特征（全因果、无标签）
FEATURE_KEYS = (
    "n_days",
    "qqq_day_ret_mean",
    "qqq_day_ret_std",
    "qqq_day_ret_abs_mean",
    "qqq_intraday_range_mean",
    "qqq_intraday_range_std",
    "qqq_gap_abs_mean",
    "qqq_up_day_frac",
    "qqq_trend_path_abs",  # |收盘路径线性拟合斜率| * bars
    "qqq_trend_path_r2",
    "qqq_vol_mean_log",
    "vixy_level_mean",
    "vixy_level_std",
    "vixy_level_change",
    "vixy_day_ret_std",
    "vix_z_mean",
    "vix_z_std",
    "vix_rev30_mean",
)


@dataclass(frozen=True)
class WeekWindow:
    start: date
    end: date
    days: tuple[date, ...]
    label: str

    @property
    def n_days(self) -> int:
        return len(self.days)


def _expand(p: str | Path) -> Path:
    return Path(os.path.expanduser(str(p))).resolve()


def _parse_query(s: str) -> tuple[date, date]:
    s = s.strip()
    if ":" in s:
        a, b = s.split(":", 1)
        return date.fromisoformat(a.strip()), date.fromisoformat(b.strip())
    d = date.fromisoformat(s)
    return d, d


def _ny_ts(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s, utc=False)
    if getattr(ts.dt, "tz", None) is None:
        return ts.dt.tz_localize("America/New_York")
    return ts.dt.tz_convert("America/New_York")


def load_symbol_1m(root: Path, symbol: str, start: date, end: date) -> pd.DataFrame:
    """按月 parquet 加载 [start, end] 交易分钟。"""
    base = root / symbol / "regular" / "09:30-16:00" / "1min"
    if not base.is_dir():
        raise FileNotFoundError(f"missing 1min dir: {base}")
    months = pd.period_range(start=start.replace(day=1), end=end.replace(day=1), freq="M")
    frames: list[pd.DataFrame] = []
    for per in months:
        fp = base / f"{per.strftime('%Y-%m')}.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        if "timestamp" not in df.columns:
            raise ValueError(f"{fp} missing timestamp")
        df = df.copy()
        df["timestamp"] = _ny_ts(df["timestamp"])
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no monthly files for {symbol} in {base} covering {start}..{end}")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    d = out["timestamp"].dt.date
    out = out[(d >= start) & (d <= end)].reset_index(drop=True)
    return out


def _session_day_frame(qqq: pd.DataFrame, vixy: pd.DataFrame) -> pd.DataFrame:
    """聚合到交易日一行（仅用当日已实现信息）。"""
    q = qqq.copy()
    q["day"] = q["timestamp"].dt.date
    v = vixy.copy()
    v["day"] = v["timestamp"].dt.date

    def _day_stats(g: pd.DataFrame) -> pd.Series:
        o = float(g["open"].iloc[0])
        h = float(g["high"].max())
        l = float(g["low"].min())
        c = float(g["close"].iloc[-1])
        vol = float(pd.to_numeric(g["volume"], errors="coerce").fillna(0).sum())
        # 路径趋势：分钟 close 对时间回归
        closes = pd.to_numeric(g["close"], errors="coerce").to_numpy(dtype=float)
        closes = closes[np.isfinite(closes)]
        slope_abs = 0.0
        r2 = 0.0
        if len(closes) >= 30 and o > 0:
            x = np.arange(len(closes), dtype=float)
            x = (x - x.mean()) / (x.std() + 1e-12)
            y = (closes / o) - 1.0
            y = y - y.mean()
            denom = float(np.dot(x, x))
            if denom > 0:
                beta = float(np.dot(x, y) / denom)
                yhat = beta * x
                ss_res = float(np.dot(y - yhat, y - yhat))
                ss_tot = float(np.dot(y, y)) + 1e-18
                r2 = max(0.0, 1.0 - ss_res / ss_tot)
                slope_abs = abs(beta)
        return pd.Series(
            {
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "range_pct": (h - l) / o if o > 0 else np.nan,
                "day_ret": (c / o - 1.0) if o > 0 else np.nan,
                "volume": vol,
                "trend_slope_abs": slope_abs,
                "trend_r2": r2,
            }
        )

    qd = q.groupby("day", sort=True).apply(_day_stats, include_groups=False)
    qd = qd.reset_index()

    # VIXY
    def _vix_day(g: pd.DataFrame) -> pd.Series:
        o = float(g["open"].iloc[0])
        c = float(g["close"].iloc[-1])
        level = pd.to_numeric(g.get("vix_level", g["close"]), errors="coerce")
        z = pd.to_numeric(g["vix_z"], errors="coerce") if "vix_z" in g.columns else pd.Series(dtype=float)
        # 粗略 30m reversal：用 close 路径
        closes = pd.to_numeric(g["close"], errors="coerce").to_numpy(dtype=float)
        rev_vals = []
        thr = 0.0015
        for i in range(29, len(closes)):
            chunk = closes[i - 29 : i + 1]
            chunk = chunk[np.isfinite(chunk)]
            if len(chunk) < 2:
                continue
            rev = 0
            last_dir = 0
            for j in range(1, len(chunk)):
                if chunk[j - 1] == 0:
                    continue
                diff = (chunk[j] - chunk[j - 1]) / chunk[j - 1]
                if abs(diff) < thr:
                    continue
                dirc = 1 if diff > 0 else -1
                if last_dir and dirc != last_dir:
                    rev += 1
                last_dir = dirc
            rev_vals.append(rev)
        return pd.Series(
            {
                "vixy_open": o,
                "vixy_close": c,
                "vixy_day_ret": (c / o - 1.0) if o > 0 else np.nan,
                "vix_level_mean": float(level.mean()) if level.notna().any() else float(c),
                "vix_z_mean": float(z.mean()) if len(z) and z.notna().any() else np.nan,
                "vix_z_std": float(z.std()) if len(z) and z.notna().sum() > 1 else np.nan,
                "vix_rev30_mean": float(np.mean(rev_vals)) if rev_vals else np.nan,
            }
        )

    vd = v.groupby("day", sort=True).apply(_vix_day, include_groups=False).reset_index()
    out = qd.merge(vd, on="day", how="inner")
    out = out.sort_values("day").reset_index(drop=True)
    # 隔夜 gap：相对前收
    out["prev_close"] = out["close"].shift(1)
    out["gap"] = out["open"] / out["prev_close"] - 1.0
    return out


def week_vector(day_df: pd.DataFrame, days: Iterable[date]) -> dict[str, float]:
    days = tuple(days)
    sub = day_df[day_df["day"].isin(days)].copy()
    if sub.empty:
        return {k: float("nan") for k in FEATURE_KEYS}

    def _m(col: str) -> float:
        s = pd.to_numeric(sub[col], errors="coerce")
        return float(s.mean()) if s.notna().any() else float("nan")

    def _s(col: str) -> float:
        s = pd.to_numeric(sub[col], errors="coerce")
        return float(s.std(ddof=0)) if s.notna().sum() > 1 else 0.0

    rets = pd.to_numeric(sub["day_ret"], errors="coerce")
    gaps = pd.to_numeric(sub["gap"], errors="coerce")
    vol = pd.to_numeric(sub["volume"], errors="coerce").replace(0, np.nan)
    vix_lvl = pd.to_numeric(sub["vix_level_mean"], errors="coerce")

    return {
        "n_days": float(len(sub)),
        "qqq_day_ret_mean": _m("day_ret"),
        "qqq_day_ret_std": _s("day_ret"),
        "qqq_day_ret_abs_mean": float(rets.abs().mean()) if rets.notna().any() else float("nan"),
        "qqq_intraday_range_mean": _m("range_pct"),
        "qqq_intraday_range_std": _s("range_pct"),
        "qqq_gap_abs_mean": float(gaps.abs().mean()) if gaps.notna().any() else float("nan"),
        "qqq_up_day_frac": float((rets > 0).mean()) if rets.notna().any() else float("nan"),
        "qqq_trend_path_abs": _m("trend_slope_abs"),
        "qqq_trend_path_r2": _m("trend_r2"),
        "qqq_vol_mean_log": float(np.log(vol.mean())) if vol.notna().any() else float("nan"),
        "vixy_level_mean": _m("vix_level_mean"),
        "vixy_level_std": _s("vix_level_mean"),
        "vixy_level_change": float(vix_lvl.iloc[-1] - vix_lvl.iloc[0])
        if vix_lvl.notna().sum() >= 2
        else 0.0,
        "vixy_day_ret_std": _s("vixy_day_ret"),
        "vix_z_mean": _m("vix_z_mean"),
        "vix_z_std": _m("vix_z_std") if "vix_z_std" in sub.columns else _s("vix_z_mean"),
        "vix_rev30_mean": _m("vix_rev30_mean"),
    }


def iter_calendar_weeks(trading_days: list[date]) -> list[WeekWindow]:
    """按 ISO 周聚合交易日（Mon–Sun 标签，实际仅含有行情的交易日）。"""
    if not trading_days:
        return []
    buckets: dict[tuple[int, int], list[date]] = {}
    for d in trading_days:
        iso = d.isocalendar()
        key = (iso.year, iso.week)
        buckets.setdefault(key, []).append(d)
    out: list[WeekWindow] = []
    for (y, w), days in sorted(buckets.items()):
        days = tuple(sorted(days))
        out.append(
            WeekWindow(
                start=days[0],
                end=days[-1],
                days=days,
                label=f"{y}-W{w:02d}({days[0]}..{days[-1]})",
            )
        )
    return out


def iter_rolling_windows(trading_days: list[date], n_days: int) -> list[WeekWindow]:
    """滚动 N 个交易日窗（与 query 交易日数对齐时更公平）。"""
    days = sorted(trading_days)
    if n_days <= 0 or len(days) < n_days:
        return []
    out: list[WeekWindow] = []
    for i in range(0, len(days) - n_days + 1):
        chunk = tuple(days[i : i + n_days])
        out.append(
            WeekWindow(
                start=chunk[0],
                end=chunk[-1],
                days=chunk,
                label=f"roll{n_days}d({chunk[0]}..{chunk[-1]})",
            )
        )
    return out


def vector_matrix(rows: list[dict[str, float]], keys: tuple[str, ...] = FEATURE_KEYS) -> np.ndarray:
    mat = np.zeros((len(rows), len(keys)), dtype=float)
    for i, r in enumerate(rows):
        for j, k in enumerate(keys):
            v = r.get(k, float("nan"))
            mat[i, j] = float(v) if v is not None and np.isfinite(v) else np.nan
    return mat


def standardize_fit(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(mat, axis=0)
    sd = np.nanstd(mat, axis=0)
    sd = np.where(~np.isfinite(sd) | (sd < 1e-8), 1.0, sd)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    return mu, sd


def apply_standardize(mat: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    x = (mat - mu) / sd
    # 缺失维用 0（等于均值）
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    sim = float(np.dot(a, b) / (na * nb))
    return 1.0 - sim


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def months_covering(days: Iterable[date]) -> list[str]:
    return sorted({d.strftime("%Y-%m") for d in days})


def suggest_train_months(
    report: dict[str, Any],
    *,
    apply: str = "blend_on_shift",
    available: Optional[Sequence[str]] = None,
    recent_lookback: int = 2,
) -> dict[str, Any]:
    """从 match 报告解析 train months。

    apply:
      - blend_on_shift: NEAR_RECENT → 近 lookback 月；SHIFT* → 近月∪相似月
      - always_blend: 总是用 suggested_train_months_example
      - suggest_only: 返回建议但不标记 applied（调用方自行决定）
    """
    rec = report.get("recommendation") or {}
    vs = report.get("vs_recent") or {}
    recent = list(vs.get("recent_months_list") or [])
    blend = list(rec.get("suggested_train_months_example") or [])
    similar = list(rec.get("blend_similar_months") or [])
    use_default = bool(rec.get("use_default_recent_ft", True))
    shift = bool(vs.get("regime_shift_flag")) or bool(vs.get("strong_shift_flag"))

    default_recent = recent[-max(1, int(recent_lookback)) :] if recent else []
    if apply == "always_blend":
        chosen = blend or default_recent
        reason = "always_blend"
    elif apply == "suggest_only":
        chosen = blend or default_recent
        reason = "suggest_only"
    else:  # blend_on_shift
        if shift or not use_default:
            chosen = blend or (default_recent + similar[:2])
            reason = "blend_on_shift"
        else:
            chosen = default_recent
            reason = "near_recent_default"

    chosen = sorted({m for m in chosen if m})
    dropped: list[str] = []
    if available is not None:
        avail_set = set(available)
        kept = [m for m in chosen if m in avail_set]
        dropped = [m for m in chosen if m not in avail_set]
        if not kept and default_recent:
            kept = [m for m in default_recent if m in avail_set]
        chosen = kept

    return {
        "train_months": chosen,
        "train_months_csv": ",".join(chosen),
        "apply": apply,
        "reason": reason,
        "regime_shift": shift,
        "use_default_recent_ft": use_default,
        "action": rec.get("action"),
        "dropped_unavailable": dropped,
        "blend_similar_months": similar,
        "suggested_train_months_example": blend,
        "recent_months": recent,
    }


def run_regime_match(
    *,
    query: str,
    spot_root: str | Path = DEFAULT_SPOT_ROOT,
    history_start: str = "2024-01-01",
    top: int = 15,
    recent_months: int = 2,
    mode: str = "rolling",
    metric: str = "cosine",
    out_dir: Optional[str | Path] = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """执行周状态匹配，返回完整 report（可供 weekly_finetune 调用）。"""
    q0, q1 = _parse_query(query)
    if q1 < q0:
        raise ValueError("query end < start")
    hist_start = date.fromisoformat(history_start)
    load_start = min(hist_start - timedelta(days=14), q0 - timedelta(days=14))
    load_end = max(q1, date.today())

    root = _expand(spot_root)
    if not quiet:
        print(f"[load] QQQ/VIXY 1m from {root}  {load_start}..{load_end}")
    qqq = load_symbol_1m(root, "QQQ", load_start, load_end)
    vixy = load_symbol_1m(root, "VIXY", load_start, load_end)
    day_df = _session_day_frame(qqq, vixy)
    all_days = [d for d in day_df["day"].tolist() if isinstance(d, date)]

    query_days = tuple(d for d in all_days if q0 <= d <= q1)
    if len(query_days) < 2:
        raise ValueError(f"query window has <2 trading days in data: {query_days}")
    query_win = WeekWindow(
        start=query_days[0],
        end=query_days[-1],
        days=query_days,
        label=f"QUERY({query_days[0]}..{query_days[-1]})",
    )
    qvec = week_vector(day_df, query_days)
    if not quiet:
        print(f"[query] {query_win.label} n_days={query_win.n_days}")

    hist_days = [d for d in all_days if hist_start <= d < query_days[0]]
    if mode == "calendar_week":
        cands = iter_calendar_weeks(hist_days)
        cands = [w for w in cands if w.n_days >= max(2, query_win.n_days - 2)]
    else:
        cands = iter_rolling_windows(hist_days, n_days=query_win.n_days)
    if not cands:
        raise ValueError("no candidate windows in history")

    cand_vecs = [week_vector(day_df, w.days) for w in cands]
    mat = vector_matrix(cand_vecs)
    mu, sd = standardize_fit(mat)
    z_hist = apply_standardize(mat, mu, sd)
    z_q = apply_standardize(vector_matrix([qvec]), mu, sd)[0]

    dist_fn = cosine_distance if metric == "cosine" else euclidean
    scores = np.array([dist_fn(z_q, z_hist[i]) for i in range(len(cands))], dtype=float)
    order = np.argsort(scores)

    recent_cut = (query_days[0].replace(day=1) - pd.DateOffset(months=int(recent_months))).date()
    recent_idx = [i for i, w in enumerate(cands) if w.end >= recent_cut]
    if recent_idx:
        recent_centroid = z_hist[recent_idx].mean(axis=0)
        dist_recent = dist_fn(z_q, recent_centroid)
        recent_best_i = min(recent_idx, key=lambda i: scores[i])
        dist_recent_best = float(scores[recent_best_i])
    else:
        dist_recent = float("nan")
        recent_best_i = -1
        dist_recent_best = float("nan")

    pct_worse_than_hist = float((scores < dist_recent).mean()) if np.isfinite(dist_recent) else float("nan")

    top_n = min(int(top), len(order))
    top_rows: list[dict[str, Any]] = []
    for rank, i in enumerate(order[:top_n], start=1):
        w = cands[i]
        top_rows.append(
            {
                "rank": rank,
                "distance": float(scores[i]),
                "label": w.label,
                "start": w.start.isoformat(),
                "end": w.end.isoformat(),
                "n_days": w.n_days,
                "months": months_covering(w.days),
                "features": {k: cand_vecs[i][k] for k in FEATURE_KEYS},
            }
        )

    med_dist = float(np.median(scores))
    p25 = float(np.percentile(scores, 25))
    regime_shift_flag = bool(np.isfinite(dist_recent) and dist_recent > med_dist)
    strong_shift = bool(np.isfinite(dist_recent) and dist_recent > float(np.percentile(scores, 75)))

    month_votes: dict[str, float] = {}
    for r in top_rows:
        wgt = 1.0 / (1e-6 + float(r["distance"]))
        for m in r["months"]:
            month_votes[m] = month_votes.get(m, 0.0) + wgt
    suggested_months = sorted(month_votes.keys(), key=lambda m: -month_votes[m])[:4]

    recent_months_list = sorted(
        {d.strftime("%Y-%m") for d in all_days if recent_cut <= d < query_days[0]}
    )
    blend_example = (
        sorted(set(recent_months_list[-2:] + suggested_months[:2]))
        if recent_months_list
        else suggested_months[:3]
    )

    report: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "query": {
            "start": query_days[0].isoformat(),
            "end": query_days[-1].isoformat(),
            "n_days": query_win.n_days,
            "label": query_win.label,
            "features": qvec,
        },
        "settings": {
            "spot_root": str(root),
            "history_start": hist_start.isoformat(),
            "mode": mode,
            "metric": metric,
            "recent_months": int(recent_months),
            "n_candidates": len(cands),
            "feature_keys": list(FEATURE_KEYS),
        },
        "vs_recent": {
            "recent_months_list": recent_months_list,
            "distance_to_recent_centroid": dist_recent,
            "distance_to_best_recent_window": dist_recent_best,
            "best_recent_window": cands[recent_best_i].label if recent_best_i >= 0 else None,
            "hist_distance_median": med_dist,
            "hist_distance_p25": p25,
            "hist_distance_p75": float(np.percentile(scores, 75)),
            "fraction_hist_closer_than_recent_centroid": pct_worse_than_hist,
            "regime_shift_flag": regime_shift_flag,
            "strong_shift_flag": strong_shift,
        },
        "top_matches": top_rows,
        "recommendation": {
            "note": "诊断建议，不自动改生产权重。相似≠可赚；用于 FT 采样混合或压力测试。",
            "use_default_recent_ft": not regime_shift_flag,
            "blend_similar_months": suggested_months,
            "suggested_train_months_example": blend_example,
            "action": (
                "NEAR_RECENT: 近月中心仍较近，可维持近 2 月微调，并用 Top-K 做压力测试"
                if not regime_shift_flag
                else (
                    "SHIFT: 本周更不像近月中心；建议近月 ∪ Top-K 相似月混合微调，或先用相似月做 replay 压力测试"
                    if not strong_shift
                    else "STRONG_SHIFT: 距近月中心偏远；优先审阅 Top-K 相似段再决定 FT 窗，避免只吃近 2 月"
                )
            ),
        },
    }
    report["train_months_suggestion"] = suggest_train_months(
        report, apply="blend_on_shift", recent_lookback=int(recent_months)
    )

    if not quiet:
        print("\n=== query features (selected) ===")
        for k in (
            "qqq_day_ret_std",
            "qqq_intraday_range_mean",
            "qqq_trend_path_r2",
            "vixy_level_mean",
            "vix_z_mean",
            "vix_rev30_mean",
        ):
            v = qvec.get(k, float("nan"))
            print(f"  {k}: {v:.6g}" if np.isfinite(v) else f"  {k}: nan")

        vr = report["vs_recent"]
        print("\n=== vs recent lookback ===")
        print(f"  recent months: {vr['recent_months_list']}")
        print(f"  dist(query, recent_centroid)={vr['distance_to_recent_centroid']:.4f}")
        print(f"  dist(query, best_recent_window)={vr['distance_to_best_recent_window']:.4f}")
        print(
            f"  hist median/p25/p75={vr['hist_distance_median']:.4f}/"
            f"{vr['hist_distance_p25']:.4f}/{vr['hist_distance_p75']:.4f}"
        )
        print(f"  regime_shift={vr['regime_shift_flag']}  strong_shift={vr['strong_shift_flag']}")
        print(f"\n=== action ===\n  {report['recommendation']['action']}")
        print(f"  blend_similar_months={report['recommendation']['blend_similar_months']}")
        print(
            f"  suggested_train_months_example="
            f"{report['recommendation']['suggested_train_months_example']}"
        )
        print(
            f"  train_months_suggestion(blend_on_shift)="
            f"{report['train_months_suggestion']['train_months_csv']}"
        )

        print(f"\n=== top {top_n} matches ({metric}) ===")
        for r in top_rows:
            print(
                f"  #{r['rank']:02d} dist={r['distance']:.4f}  "
                f"{r['start']}..{r['end']}  months={r['months']}"
            )

    if out_dir:
        od = _expand(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        (od / "regime_match.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        pd.DataFrame(
            [
                {
                    "rank": r["rank"],
                    "distance": r["distance"],
                    "start": r["start"],
                    "end": r["end"],
                    "n_days": r["n_days"],
                    "months": ",".join(r["months"]),
                    "label": r["label"],
                }
                for r in top_rows
            ]
        ).to_csv(od / "top_matches.csv", index=False)
        if not quiet:
            print(f"\n[wrote] {od / 'regime_match.json'}")
            print(f"[wrote] {od / 'top_matches.csv'}")

    return report


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Match query week regime to historical weeks")
    ap.add_argument("--query", required=True, help="YYYY-MM-DD or start:end (inclusive)")
    ap.add_argument(
        "--spot-root",
        default=str(DEFAULT_SPOT_ROOT),
        help="spnq_train_resampled root",
    )
    ap.add_argument("--history-start", default="2024-01-01", help="history search start date")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--recent-months", type=int, default=2, help="compare vs last N calendar months centroid")
    ap.add_argument(
        "--mode",
        choices=("calendar_week", "rolling"),
        default="rolling",
        help="rolling=同长度交易日窗（默认，更公平）；calendar_week=ISO 周",
    )
    ap.add_argument("--metric", choices=("cosine", "euclidean"), default="cosine")
    ap.add_argument("--out", default="", help="output dir for json/csv (optional)")
    ap.add_argument(
        "--emit-train-months",
        action="store_true",
        help="stdout 仅打印建议 train months CSV（配合 --quiet / --apply）",
    )
    ap.add_argument(
        "--apply",
        choices=("blend_on_shift", "always_blend", "suggest_only"),
        default="blend_on_shift",
        help="如何从报告选出 train months（默认仅 SHIFT 时混合）",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    try:
        report = run_regime_match(
            query=args.query,
            spot_root=args.spot_root,
            history_start=args.history_start,
            top=args.top,
            recent_months=args.recent_months,
            mode=args.mode,
            metric=args.metric,
            out_dir=args.out or None,
            quiet=bool(args.quiet or args.emit_train_months),
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    suggestion = suggest_train_months(
        report, apply=args.apply, recent_lookback=int(args.recent_months)
    )
    report["train_months_suggestion"] = suggestion

    if args.emit_train_months:
        print(suggestion["train_months_csv"])
        return 0

    if args.quiet:
        print(json.dumps(suggestion, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
