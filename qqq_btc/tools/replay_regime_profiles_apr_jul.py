#!/usr/bin/env python3
"""4–7 月：baseline vs 日频 regime rule-profile 切换对照。

因果选 profile（日前 lookback=5 交易日）：
  OPEN_DEFENSE  : vix_z >= 0.5 → 10:30 + PUT q10
  CHOP_NO_TRADE : 压缩 range + 弱上涨占比 → 当日禁新开（May/Jun 提炼）
  TREND_PUT_OK  : 默认 09:45 + CALL-only q10

用法:
  python qqq_btc/tools/replay_regime_profiles_apr_jul.py
  python qqq_btc/tools/replay_regime_profiles_apr_jul.py --skip-build
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from qqq_btc.common.replay_harness import run_strict_replay
from qqq_btc.common.rule_profiles import (
    apply_rule_profile,
    assign_daily_profiles,
    load_rule_profiles,
    select_profile_name_vx,
)
from qqq_btc.qqq import config as qcfg
from qqq_btc.tools.match_week_regime import load_symbol_1m

PY = os.environ.get("PYTHON", str(Path.home() / "anaconda3/envs/ibkr/bin/python"))
SPOT_ROOT = Path.home() / "train_data/spnq_train_resampled"
OUT_DIR = REPO / "qqq_btc/results/regime_rule_profiles_apr_jul"
DEFAULT_VX_TERM = Path(
    "/mnt/s990/data/raw_1m/vix_futures_databento/vx_term_structure_1d.parquet"
)


def _month_paths(ym: str) -> dict[str, Path]:
    m = ym[5:7]
    name = {"04": "april", "05": "may", "06": "june", "07": "july"}[m]
    exp = Path.home() / f"train_data/{name}_v4_old_lock"
    return {
        "ym": ym,
        "exp": exp,
        "infer": REPO / f"qqq_btc/results/v4_{ym}_old_lock/infer/test_infer.parquet"
        if m != "07"
        else REPO / "qqq_btc/results/v4_jul_w1_fixed5m_infer/test_infer.parquet",
        "raw1": exp / f"quote_features_raw/QQQ/regular/09:30-16:00/1min/{ym}.parquet"
        if m != "07"
        else Path.home()
        / "train_data/july_w1_v4_honest_openwin/quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-07.parquet",
        "opt1m": exp / f"options_1m_{ym}" if m != "07" else Path.home() / "train_data/july_w1_v4_honest_openwin/options_1m",
    }


def ensure_month(ym: str, *, skip_build: bool) -> dict[str, Path]:
    paths = _month_paths(ym)
    if ym == "2026-07":
        if not paths["infer"].exists():
            raise FileNotFoundError(f"missing july infer: {paths['infer']}")
        return paths
    # alias existing april/june result dirs
    if ym == "2026-04":
        alt = REPO / "qqq_btc/results/v4_april2026_old_lock/infer/test_infer.parquet"
        if alt.exists():
            paths["infer"] = alt
        if not paths["raw1"].exists():
            paths["raw1"] = Path.home() / "train_data/april_v4_old_lock/quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-04.parquet"
    if ym == "2026-06":
        alt = REPO / "qqq_btc/results/v4_june2026_old_lock/infer/test_infer.parquet"
        if alt.exists():
            paths["infer"] = alt
        if not paths["raw1"].exists():
            paths["raw1"] = Path.home() / "train_data/june_v4_old_lock/quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-06.parquet"

    if paths["infer"].exists() and paths["raw1"].exists():
        return paths
    if skip_build:
        raise FileNotFoundError(f"missing {ym} infer/raw1 and --skip-build set")
    cmd = ["bash", str(REPO / "qqq_btc/tools/build_v4_old_lock_month.sh"), ym]
    print("building", ym, flush=True)
    subprocess.check_call(cmd, cwd=str(REPO))
    # refresh paths after build
    paths = _month_paths(ym)
    if ym == "2026-05":
        # build script writes to v4_2026-05_old_lock or may_v4
        cand = [
            REPO / "qqq_btc/results/v4_2026-05_old_lock/infer/test_infer.parquet",
            Path.home() / "train_data/may_v4_old_lock",
        ]
        inf = REPO / "qqq_btc/results/v4_2026-05_old_lock/infer/test_infer.parquet"
        if inf.exists():
            paths["infer"] = inf
        paths["raw1"] = Path.home() / "train_data/may_v4_old_lock/quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-05.parquet"
        paths["exp"] = Path.home() / "train_data/may_v4_old_lock"
    if not paths["infer"].exists() or not paths["raw1"].exists():
        raise FileNotFoundError(f"build finished but missing {paths}")
    return paths


def attach_causal_put_gate(inf: pd.DataFrame, raw1_path: Path) -> pd.DataFrame:
    out = inf.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.drop(columns=["put_gate", "vix_proxy_close"], errors="ignore")
    raw1 = pd.read_parquet(
        raw1_path, columns=["timestamp", "vix_level", "vix_proxy_close"]
    )
    raw1["timestamp"] = pd.to_datetime(raw1["timestamp"], utc=True)
    raw1 = raw1.sort_values("timestamp").drop_duplicates("timestamp")
    s = raw1[["timestamp", "vix_level"]].copy()
    s["timestamp"] = s["timestamp"] + pd.Timedelta(minutes=1)
    m = pd.merge_asof(
        out[["timestamp"]].reset_index(drop=True),
        s.rename(columns={"vix_level": "put_gate"}),
        on="timestamp",
        direction="backward",
    )
    out["put_gate"] = m["put_gate"].to_numpy()
    # 方向确认使用当前已完成分钟的 VIXY proxy；不沿用 put_gate 的 +1min
    # 安全位移，否则会把 15m 收益再额外滞后一根。
    out = pd.merge_asof(
        out.sort_values("timestamp"),
        raw1[["timestamp", "vix_proxy_close"]],
        on="timestamp",
        direction="backward",
    ).sort_index()
    return out


def daily_from_trades(trades: pd.DataFrame) -> list[dict[str, Any]]:
    if trades is None or len(trades) == 0 or "entry_ts" not in trades.columns:
        return []
    t = trades.copy()
    t["entry_ny"] = pd.to_datetime(t["entry_ts"]).dt.tz_convert("America/New_York")
    t["date"] = t["entry_ny"].dt.strftime("%Y-%m-%d")
    daily = []
    eq = 1.0
    for date_s, g in t.groupby("date", sort=True):
        day_eq = 1.0
        for _, trade in g.iterrows():
            frac = float(trade.get("position_frac", 0.25))
            r = float(trade["net_return"])
            day_eq *= 1 + frac * r
            eq *= 1 + frac * r
        daily.append(
            {
                "date": date_s,
                "n": int(len(g)),
                "day_acct25": float(day_eq - 1),
                "cum_acct25": float(eq - 1),
                "hit": float((g["net_return"] > 0).mean()),
                "legs": g["leg"].value_counts().to_dict() if "leg" in g.columns else {},
            }
        )
    return daily


def replay_with_cfg(df: pd.DataFrame, cfg) -> dict[str, Any]:
    res = run_strict_replay(
        df,
        qcfg.FILL_MODEL,
        cfg,
        qcfg.EXIT_RAILS,
        edge_col="net_edge",
        edge_q10_col="net_edge_q10",
        call_edge_col="call_net_edge",
        put_edge_col="put_net_edge",
        put_gate_col="put_gate",
    )
    s = res.summary(position_frac=0.25)
    trades = res.trades_frame()
    return {
        "acct25": float(s["total_net_return"]),
        "trades": int(s["trades"]),
        "hit": float(s.get("hit_rate") or 0),
        "pf": float(s["profit_factor"]) if s.get("profit_factor") is not None else None,
        "legs": s.get("trades_by_leg"),
        "worst": float(s["worst_trade"]) if s.get("worst_trade") is not None else None,
        "daily": daily_from_trades(trades),
        "trades_frame": trades,
    }


def early_drawdown(daily: list[dict[str, Any]], first_n_days: int = 4) -> float | None:
    if not daily:
        return None
    # min cum over first n trading days present
    chunk = daily[:first_n_days]
    if not chunk:
        return None
    return float(min(d["cum_acct25"] for d in chunk))


def _mask_chop_days(df: pd.DataFrame, chop_days: set[date]) -> pd.DataFrame:
    """把 CHOP 日的 edge 置 NaN，使单次 replay 内禁开且不打断跨日状态。"""
    if not chop_days:
        return df
    out = df.copy()
    if "date" not in out.columns:
        out["date"] = pd.to_datetime(out["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.date
    m = out["date"].isin(chop_days)
    for c in (
        "net_edge",
        "net_edge_q10",
        "call_net_edge",
        "put_net_edge",
        "call_net_edge_q10",
        "put_net_edge_q10",
    ):
        if c in out.columns:
            out.loc[m, c] = float("nan")
    return out


def _contiguous_non_chop_segments(
    day_profiles: dict[date, dict[str, Any]],
) -> list[tuple[str, list[date], set[date]]]:
    """按 OPEN/TREND 连续段切分；段内 CHOP 日单独收集，稍后 mask。

    返回 (active_profile, all_days_in_span, chop_days_in_span)。
    CHOP 日并入相邻 TREND/OPEN 段的日历跨度，但会 mask。
    独立前缀/后缀纯 CHOP 段用 CHOP_NO_TRADE。
    """
    days = sorted(day_profiles.keys())
    if not days:
        return []

    # 先把每天映射到「有效交易 profile」：CHOP 暂记为 None
    effective = []
    for d in days:
        p = day_profiles[d]["profile"]
        effective.append((d, None if p == "CHOP_NO_TRADE" else p))

    # 前向填充：CHOP 继承前一个非 CHOP profile；若开头全是 CHOP，标 CHOP_NO_TRADE
    filled: list[tuple[date, str]] = []
    last = "CHOP_NO_TRADE"
    for d, p in effective:
        if p is None:
            filled.append((d, last))
        else:
            last = p
            filled.append((d, p))
    # 若开头被填成 CHOP_NO_TRADE，保持；中间 CHOP 已并入 last

    segs: list[tuple[str, list[date], set[date]]] = []
    cur_prof = filled[0][1]
    cur_days = [filled[0][0]]
    cur_chop: set[date] = set()
    if day_profiles[filled[0][0]]["profile"] == "CHOP_NO_TRADE":
        cur_chop.add(filled[0][0])

    for d, p in filled[1:]:
        is_chop = day_profiles[d]["profile"] == "CHOP_NO_TRADE"
        if p == cur_prof:
            cur_days.append(d)
            if is_chop:
                cur_chop.add(d)
        else:
            segs.append((cur_prof, cur_days, cur_chop))
            cur_prof = p
            cur_days = [d]
            cur_chop = {d} if is_chop else set()
    segs.append((cur_prof, cur_days, cur_chop))
    return segs


def regime_profile_stitch(
    df: pd.DataFrame,
    day_profiles: dict[date, dict[str, Any]],
    profiles_cfg: dict[str, Any],
    base_cfg,
) -> dict[str, Any]:
    """Regime 切换 replay（与 full-month baseline 同口径：段内跨日状态保留）。

    - 单一 profile：整月一次
    - OPEN/TREND 切换：按连续段切
    - CHOP：段内 mask edge（不单独切段，避免状态重置伪增益）
    """
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    work["date"] = work["timestamp"].dt.tz_convert("America/New_York").dt.date

    uniq = sorted({day_profiles[d]["profile"] for d in day_profiles})
    day_reports = []
    for d in sorted(day_profiles.keys()):
        meta = day_profiles[d]
        day_reports.append(
            {
                "date": d.isoformat(),
                "week_start": meta.get("week_start"),
                "profile": meta["profile"],
                "vix_z_mean_lookback": meta.get("vix_z_mean_lookback"),
                "qqq_up_frac_lookback": meta.get("qqq_up_frac_lookback"),
                "qqq_range_mean_lookback": meta.get("qqq_range_mean_lookback"),
                "lookback": meta.get("lookback"),
            }
        )

    chop_only = uniq == ["CHOP_NO_TRADE"]
    non_chop = [p for p in uniq if p != "CHOP_NO_TRADE"]
    if len(uniq) == 1 or (len(non_chop) == 1 and "CHOP_NO_TRADE" in uniq and not chop_only):
        # 整月同一交易 profile（可含 CHOP mask）
        prof = non_chop[0] if non_chop else "CHOP_NO_TRADE"
        chop_days = {d for d, m in day_profiles.items() if m["profile"] == "CHOP_NO_TRADE"}
        cfg = apply_rule_profile(base_cfg, prof, profiles_cfg=profiles_cfg)
        part = replay_with_cfg(_mask_chop_days(work, chop_days), cfg)
        part.pop("trades_frame", None)
        part["weeks"] = day_reports
        part["segments"] = [
            {
                "profile": prof,
                "days": [d.isoformat() for d in sorted(day_profiles)],
                "chop_masked": sorted(x.isoformat() for x in chop_days),
            }
        ]
        part["early4_min_cum"] = early_drawdown(part["daily"], 4)
        part["single_profile_fastpath"] = True
        return part

    all_trades = []
    eq = 1.0
    segments = []
    for prof, days, chop_days in _contiguous_non_chop_segments(day_profiles):
        cfg = apply_rule_profile(base_cfg, prof, profiles_cfg=profiles_cfg)
        sub = work[work["date"].isin(days)].reset_index(drop=True)
        if sub.empty:
            continue
        sub = _mask_chop_days(sub, chop_days)
        part = replay_with_cfg(sub, cfg)
        seg_eq = 1.0
        tf = part["trades_frame"]
        if tf is not None and len(tf):
            tf = tf.copy().sort_values("entry_ts")
            for _, trade in tf.iterrows():
                frac = float(trade.get("position_frac", 0.25))
                r = float(trade["net_return"])
                seg_eq *= 1 + frac * r
                eq *= 1 + frac * r
            all_trades.append(tf)
        segments.append(
            {
                "profile": prof,
                "days": [d.isoformat() for d in days],
                "chop_masked": sorted(x.isoformat() for x in chop_days),
                "acct25_seg": float(seg_eq - 1),
                "trades": part["trades"],
                "legs": part["legs"],
            }
        )

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    daily = daily_from_trades(trades)
    return {
        "acct25": float(eq - 1),
        "trades": int(len(trades)) if len(trades) else 0,
        "hit": float((trades["net_return"] > 0).mean()) if len(trades) else 0.0,
        "legs": trades["leg"].value_counts().to_dict() if len(trades) and "leg" in trades.columns else {},
        "daily": daily,
        "weeks": day_reports,
        "segments": segments,
        "early4_min_cum": early_drawdown(daily, 4),
        "single_profile_fastpath": False,
    }


def run_month(
    ym: str,
    paths: dict[str, Path],
    vixy: pd.DataFrame,
    qqq: pd.DataFrame,
    profiles_cfg: dict[str, Any],
    calendar_days: list[date],
    vx_term: pd.DataFrame | None = None,
    *,
    selector_source: str = "vixy",
    put_quarantine_loss: float | None = None,
    put_quarantine_vix_z_max: float | None = None,
    put_quarantine_vx_slope_min: float | None = None,
    base_replay_cfg=None,
) -> dict[str, Any]:
    inf = attach_causal_put_gate(pd.read_parquet(paths["infer"]), paths["raw1"])
    inf["timestamp"] = pd.to_datetime(inf["timestamp"], utc=True)
    trading_days = sorted(inf["timestamp"].dt.tz_convert("America/New_York").dt.date.unique())

    # 新入口可传入由 strategy profile 物化的 cfg；旧研究入口保持原口径。
    base = (
        base_replay_cfg
        if base_replay_cfg is not None
        else replace(qcfg.LIVE_REPLAY, edge_q10_floor=-0.2)
    )
    baseline_cfg = apply_rule_profile(base, "TREND_PUT_OK", profiles_cfg=profiles_cfg)
    # true default baseline = current production-like (TREND_PUT_OK equals reverted gates)
    base_res = replay_with_cfg(inf, baseline_cfg)
    base_res["early4_min_cum"] = early_drawdown(base_res["daily"], 4)
    base_res.pop("trades_frame", None)

    day_profiles = assign_daily_profiles(
        trading_days,
        vixy,
        qqq,
        profiles_cfg=profiles_cfg,
        calendar_days=calendar_days,
    )
    # 每日值只由日前 lookback 计算；ReplaySession 在日切时读取，绝不使用当日未来。
    inf["regime_vix_z"] = inf["timestamp"].dt.tz_convert(
        "America/New_York"
    ).dt.date.map(
        {d: day_profiles[d].get("vix_z_mean_lookback") for d in trading_days}
    )
    vx_slope_by_day: dict[date, float] = {}
    if vx_term is not None and not vx_term.empty:
        vx = vx_term.copy()
        vx["source_date"] = pd.to_datetime(vx["date"], utc=True).dt.date
        vx = vx.sort_values("source_date")
        # Databento ohlcv-1d 是 UTC 日桶；source_date 日桶在次日 00:00 UTC
        # 完成。QQQ 当日开盘前仅使用 source_date < trading_day 的最后一条。
        for d in trading_days:
            prior = vx.loc[vx["source_date"] < d]
            if prior.empty:
                continue
            row = prior.iloc[-1]
            slope = float(row["vx_curve_slope"])
            vx_slope_by_day[d] = slope
            day_profiles[d]["vx_source_date"] = row["source_date"].isoformat()
            day_profiles[d]["vx_curve_slope"] = slope
            day_profiles[d]["vx_curve_slope_z63"] = float(
                row["vx_curve_slope_z63"]
            )
            day_profiles[d]["vx_cm30_level_z63"] = float(
                row["vx_cm30_level_z63"]
            )
    if selector_source == "off":
        for d in trading_days:
            day_profiles[d]["profile"] = "TREND_PUT_OK"
            day_profiles[d]["selector_source"] = "off"
    elif selector_source == "vx":
        for d in trading_days:
            meta = day_profiles[d]
            meta["profile"] = select_profile_name_vx(
                float(meta.get("vx_curve_slope", float("nan"))),
                qqq_up_frac=float(
                    meta.get("qqq_up_frac_lookback", float("nan"))
                ),
                qqq_range_mean=float(
                    meta.get("qqq_range_mean_lookback", float("nan"))
                ),
                profiles_cfg=profiles_cfg,
            )
    inf["vx_curve_slope"] = inf["timestamp"].dt.tz_convert(
        "America/New_York"
    ).dt.date.map(vx_slope_by_day)
    regime_base = replace(
        base,
        next_day_put_quarantine_loss=put_quarantine_loss,
        next_day_put_quarantine_vix_z_max=put_quarantine_vix_z_max,
        next_day_put_quarantine_vx_slope_min=put_quarantine_vx_slope_min,
    )
    regime_res = regime_profile_stitch(
        inf, day_profiles, profiles_cfg, regime_base
    )
    # also always-OPEN_DEFENSE for oracle upper bounds
    always_def = replay_with_cfg(inf, apply_rule_profile(base, "OPEN_DEFENSE", profiles_cfg=profiles_cfg))
    always_def["early4_min_cum"] = early_drawdown(always_def["daily"], 4)
    always_def.pop("trades_frame", None)

    profile_counts: dict[str, int] = {}
    for meta in day_profiles.values():
        profile_counts[meta["profile"]] = profile_counts.get(meta["profile"], 0) + 1

    return {
        "ym": ym,
        "n_days": len(trading_days),
        "profile_day_counts": profile_counts,
        "day_profiles": {
            d.isoformat(): day_profiles[d] for d in trading_days
        },
        "baseline_TREND_PUT_OK": base_res,
        "regime_daily_switch": {
            k: v for k, v in regime_res.items() if k != "trades_frame"
        },
        # 兼容旧 summary 字段名
        "regime_weekly_switch": {
            k: v for k, v in regime_res.items() if k != "trades_frame"
        },
        "always_OPEN_DEFENSE": always_def,
        "delta_regime_vs_baseline_pp": (
            regime_res["acct25"] - base_res["acct25"]
        )
        * 100,
        "early4_regime_vs_baseline": {
            "baseline": base_res.get("early4_min_cum"),
            "regime": regime_res.get("early4_min_cum"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--months", default="2026-04,2026-05,2026-06,2026-07")
    ap.add_argument(
        "--selector-source",
        choices=("vixy", "vx"),
        default="vixy",
        help="profile selector 数据源；默认保持旧 VIXY 口径",
    )
    ap.add_argument(
        "--put-quarantine-loss",
        type=float,
        default=None,
        help="前日 PUT sleeve acct 贡献触发阈值，例如 -0.02；默认关闭",
    )
    ap.add_argument(
        "--put-quarantine-vix-z-max",
        type=float,
        default=None,
        help="仅当日因果 lookback vix_z 不高于该值时执行 PUT quarantine",
    )
    ap.add_argument(
        "--put-quarantine-vx-slope-min",
        type=float,
        default=None,
        help="仅当前一已完成 VX 日线的 VX2/VX1-1 不低于该值时执行",
    )
    ap.add_argument(
        "--vx-term-structure",
        type=Path,
        default=DEFAULT_VX_TERM,
    )
    args = ap.parse_args()
    months = [m.strip() for m in args.months.split(",") if m.strip()]

    profiles_cfg = load_rule_profiles()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # VIXY/QQQ history covering lookbacks into March
    vixy = load_symbol_1m(SPOT_ROOT, "VIXY", date(2026, 2, 1), date(2026, 7, 15))
    qqq = load_symbol_1m(SPOT_ROOT, "QQQ", date(2026, 2, 1), date(2026, 7, 15))
    vx_term = (
        pd.read_parquet(args.vx_term_structure)
        if args.vx_term_structure.exists()
        else None
    )
    if (
        args.selector_source == "vx"
        or args.put_quarantine_vx_slope_min is not None
    ) and vx_term is None:
        raise FileNotFoundError(
            f"VX selector/quarantine 已开启但期限结构文件不存在: "
            f"{args.vx_term_structure}"
        )
    calendar_days = sorted(
        pd.to_datetime(vixy["timestamp"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.date.unique()
        .tolist()
    )

    results = []
    for ym in months:
        print(f"\n===== {ym} =====", flush=True)
        paths = ensure_month(ym, skip_build=args.skip_build)
        print("infer", paths["infer"], flush=True)
        print("raw1", paths["raw1"], flush=True)
        row = run_month(
            ym,
            paths,
            vixy,
            qqq,
            profiles_cfg,
            calendar_days,
            vx_term,
            selector_source=args.selector_source,
            put_quarantine_loss=args.put_quarantine_loss,
            put_quarantine_vix_z_max=args.put_quarantine_vix_z_max,
            put_quarantine_vx_slope_min=args.put_quarantine_vx_slope_min,
        )
        results.append(row)
        b = row["baseline_TREND_PUT_OK"]
        r = row["regime_daily_switch"]
        d = row["always_OPEN_DEFENSE"]
        print(
            f"  baseline  acct25={b['acct25']*100:+.2f}% trades={b['trades']} early4_min={b.get('early4_min_cum')}"
        )
        print(
            f"  regime    acct25={r['acct25']*100:+.2f}% trades={r['trades']} early4_min={r.get('early4_min_cum')} "
            f"delta={row['delta_regime_vs_baseline_pp']:+.1f}pp profiles={row['profile_day_counts']}"
        )
        print(
            f"  alwaysDEF acct25={d['acct25']*100:+.2f}% trades={d['trades']} early4_min={d.get('early4_min_cum')}"
        )
        chop_days = [x for x in (r.get("weeks") or []) if x.get("profile") == "CHOP_NO_TRADE"]
        open_days = [x for x in (r.get("weeks") or []) if x.get("profile") == "OPEN_DEFENSE"]
        if open_days:
            print(f"  OPEN_DEFENSE days ({len(open_days)}): " + ", ".join(x["date"] for x in open_days))
        if chop_days:
            print(f"  CHOP_NO_TRADE days ({len(chop_days)}): " + ", ".join(x["date"] for x in chop_days))
        for seg in r.get("segments") or []:
            aw = seg.get("acct25_seg")
            aw_s = f"{aw*100:+.2f}%" if aw is not None else "n/a"
            print(
                f"    seg {seg['profile']} [{seg['days'][0]}..{seg['days'][-1]}] "
                f"acct={aw_s} n={seg.get('trades')} chop={seg.get('chop_masked')}"
            )

    summary = {
        "recipe": "LIVE q10=-0.2 + causal put_gate; daily regime profile switch (OPEN_DEFENSE / CHOP_NO_TRADE / TREND)",
        "selector_source": args.selector_source,
        "put_quarantine": {
            "loss": args.put_quarantine_loss,
            "vix_z_max": args.put_quarantine_vix_z_max,
            "vx_slope_min": args.put_quarantine_vx_slope_min,
            "vx_term_structure": str(args.vx_term_structure),
        },
        "profiles": profiles_cfg,
        "months": results,
        "verdict_hints": {
            "apr_early_drawdown": "看 early4_min_cum：regime/alwaysDEF 是否好于 baseline",
            "may_jun_chop": "看 CHOP_NO_TRADE 天数与 May/Jun acct 是否优于 baseline",
            "jul_keep_edge": "看 2026-07 regime 是否接近 baseline（避免被 OPEN_DEFENSE/CHOP 误杀）",
        },
    }
    if args.put_quarantine_loss is None:
        outp = OUT_DIR / (
            "summary.json"
            if args.selector_source == "vixy"
            else f"summary_selector_{args.selector_source}.json"
        )
    else:
        loss_tag = f"{abs(args.put_quarantine_loss):.3f}".replace(".", "p")
        vz_tag = (
            "none"
            if args.put_quarantine_vix_z_max is None
            else f"{abs(args.put_quarantine_vix_z_max):.3f}".replace(".", "p")
        )
        vx_tag = (
            "none"
            if args.put_quarantine_vx_slope_min is None
            else f"{abs(args.put_quarantine_vx_slope_min):.3f}".replace(".", "p")
        )
        outp = OUT_DIR / (
            f"summary_selector_{args.selector_source}_put_quarantine_"
            f"loss{loss_tag}_vz{vz_tag}_vx{vx_tag}.json"
        )
    # drop huge day_profiles from printed size? keep them
    outp.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n")
    print(f"\n[wrote] {outp}")


if __name__ == "__main__":
    main()
