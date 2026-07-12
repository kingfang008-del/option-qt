#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Honest sim 秒级快速对拍 —— 不跑 Redis 多进程、不注入任何离线答案。

复现实时流的因果链路（数据源 → 1m 聚合 → FCS pandas 特征 → vix_level），
逐层与离线 quote_features_raw 对比，直接定位"实盘会算错的地方"。

层级:
  L0  发球机数据源 1m OHLC          vs quote_features_raw OHLC
  L1  FCS finalize_1min_bar 聚合语义  (1s tick → 1m, 与发球机展开互逆性)
  L2  FCS _pandas_compute_features   vs 离线 price 特征列 (ADX/BB/...)
  L3  FCS vix_level 公式             vs 离线 vix_level 列 (VIXY)

用法:
  python qqq_btc/tools/honest_parity_fast.py --date 2026-06-26
  python qqq_btc/tools/honest_parity_fast.py --date 2026-06-26 --warmup-days 1 --json /tmp/honest_parity.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

_REPO = Path(__file__).resolve().parents[2]
_BASELINE = _REPO / "New_Pro" / "baseline_qqq"
_DAO = _BASELINE / "DAO"
for p in (str(_REPO), str(_BASELINE), str(_DAO)):
    if p not in sys.path:
        sys.path.insert(0, p)

NY_TZ = pytz.timezone("America/New_York")

OFFLINE_ROOT = Path.home() / "train_data/quote_features_raw"
PITCHER_STOCK_1S_ROOT = Path("/mnt/s990/data/raw_1s/stocks")
PITCHER_FALLBACK_ROOTS = {
    "QQQ": Path.home() / "train_data/spnq_train/QQQ",
    "VIXY": Path.home() / "train_data/spnq_train/VIXY",
}

PRICE_FEATURES = (
    "close_log_return",
    "vwap_log_return",
    "volume_ratio",
    "vwap_diff",
    "garman_klass_vol",
    "return_divergence",
    "bb_width",
    "adx_smooth_10",
    "rsi",
    "k",
    "cci",
)


# ---------------------------------------------------------------- utilities
def _ny(ts_series: pd.Series) -> pd.Series:
    s = pd.to_datetime(ts_series, utc=True)
    return s.dt.tz_convert(NY_TZ)


def load_offline_day(sym: str, date_iso: str) -> pd.DataFrame:
    """离线 quote_features_raw 单日(含全部特征列),index=NY 分钟。"""
    month = date_iso[:7]
    path = OFFLINE_ROOT / sym / "regular/09:30-16:00/1min" / f"{month}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = _ny(df["timestamp"])
    day = df[df["timestamp"].dt.date == pd.Timestamp(date_iso).date()].copy()
    return day.set_index("timestamp").sort_index()


def load_pitcher_source_1m(sym: str, date_iso: str) -> tuple[pd.DataFrame, str]:
    """
    发球机正股数据源的 1m 视图(与 redis_fused_pitcher_1s 同一优先级):
      1) raw 1s parquet → 按 FCS finalize_1min_bar 语义聚合成 1m
      2) fallback 1m parquet (直接就是 1m)
    返回 (df, source_tag)。
    """
    day_1s = PITCHER_STOCK_1S_ROOT / sym / f"{sym}_{date_iso}.parquet"
    if day_1s.exists():
        df = pd.read_parquet(day_1s)
        df["timestamp"] = _ny(df["timestamp"] if "timestamp" in df.columns else df["ts"])
        df = df.set_index("timestamp").between_time("09:30", "15:59")
        if df.empty:
            return pd.DataFrame(), "raw_1s(empty)"
        price = "close" if "close" in df.columns else "price"
        for c in ("open", "high", "low"):
            if c not in df.columns:
                df[c] = df[price]
        if "volume" not in df.columns:
            df["volume"] = 0.0
        agg = aggregate_1s_to_1m_fcs(df.reset_index())
        return agg, "raw_1s→fcs_1m"

    fb_root = PITCHER_FALLBACK_ROOTS.get(sym)
    if fb_root is None:
        return pd.DataFrame(), "none"
    month_path = fb_root / f"{date_iso[:7]}.parquet"
    if not month_path.exists():
        return pd.DataFrame(), f"fallback missing: {month_path}"
    dfm = pd.read_parquet(month_path)
    dfm["timestamp"] = _ny(dfm["timestamp"])
    day = dfm[dfm["timestamp"].dt.date == pd.Timestamp(date_iso).date()].copy()
    day = day.set_index("timestamp").between_time("09:30", "15:59").sort_index()
    cols = [c for c in ("open", "high", "low", "close", "volume", "vwap") if c in day.columns]
    return day[cols], f"fallback_1m({month_path.name})"


def aggregate_1s_to_1m_fcs(df_1s: pd.DataFrame) -> pd.DataFrame:
    """
    复刻 FCS finalize_1min_bar 的分钟聚合语义:
      o=首tick open, c=末tick close, h=max(high,close), l=min(low,close),
      v=sum(volume), vwap=Σ(close*vol)/Σvol
    """
    df = df_1s.copy()
    df["minute"] = df["timestamp"].dt.floor("min")
    wap_col = "vwap" if "vwap" in df.columns else "close"
    out_rows = []
    for minute, g in df.groupby("minute"):
        g = g.sort_values("timestamp")
        c = float(g["close"].iloc[-1])
        h = float(np.maximum(g["high"], g["close"]).max())
        l = float(np.minimum(g["low"], g["close"]).min())
        v = float(g["volume"].clip(lower=0).sum())
        wap = g[wap_col].where(g[wap_col] > 0, g["close"])
        pv = float((wap * g["volume"].clip(lower=0)).sum())
        out_rows.append(
            {
                "timestamp": minute,
                "open": float(g["open"].iloc[0]),
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
                "vwap": pv / (v + 1e-10) if v > 0 else c,
            }
        )
    return pd.DataFrame(out_rows).set_index("timestamp").sort_index()


def fcs_price_features(hist_1m: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    """在线路径: qqq_btc.common.price_features 唯一实现(FCS 同款,因果,无注入)。"""
    from qqq_btc.common.price_features import compute_price_pandas_features

    # FCS history_1min 含 stream 聚合出的 vwap 列(finalize_1min_bar),一并传入
    base = hist_1m[[c for c in ("open", "high", "low", "close", "volume", "vwap") if c in hist_1m.columns]].copy()
    return compute_price_pandas_features(base, list(features))


def fcs_vix_level_series(vixy_close: pd.Series) -> pd.Series:
    """在线路径: _compute_vix_global 的因果逐 bar 版 ((last-mean)/std, win=60, >20 bar)。

    与离线 generate_vix_level_global 的 rolling(60, min_periods=20) z-score 同公式;
    唯一差别是冷启动时 FCS 只有当日 bars —— warm(带前日历史)后两者应收敛。
    torch.std 与 pandas rolling std 同为 ddof=1。
    """
    arr = pd.to_numeric(vixy_close, errors="coerce").to_numpy(dtype=float)
    out = np.zeros(len(arr))
    for i in range(len(arr)):
        win = arr[max(0, i - 59) : i + 1]
        win = win[np.isfinite(win)]
        if len(win) > 20:
            out[i] = (win[-1] - win.mean()) / (win.std(ddof=1) + 1e-6)
    return pd.Series(out, index=vixy_close.index)


# ---------------------------------------------------------------- reports
def diff_frame(offline: pd.DataFrame, online: pd.DataFrame, cols: list[str]) -> list[dict]:
    idx = offline.index.intersection(online.index)
    rows = []
    for c in cols:
        if c not in offline.columns or c not in online.columns:
            rows.append({"col": c, "n": 0, "med": np.nan, "max": np.nan, "note": "missing"})
            continue
        a = pd.to_numeric(offline.loc[idx, c], errors="coerce")
        b = pd.to_numeric(online.loc[idx, c], errors="coerce")
        m = a.notna() & b.notna()
        if not m.any():
            rows.append({"col": c, "n": 0, "med": np.nan, "max": np.nan, "note": "no rows"})
            continue
        d = (a[m] - b[m]).abs()
        rows.append(
            {
                "col": c,
                "n": int(m.sum()),
                "med": float(d.median()),
                "max": float(d.max()),
                "argmax": str(d.idxmax()),
            }
        )
    return rows


def fmt_rows(rows: list[dict], tol_med: float) -> str:
    lines = []
    for r in rows:
        if r.get("note"):
            lines.append(f"  {r['col']:22s} -- {r['note']}")
            continue
        flag = "ok " if r["med"] <= tol_med else "GAP"
        lines.append(
            f"  {r['col']:22s} med={r['med']:.3e} max={r['max']:.3e} n={r['n']} [{flag}] worst@{r.get('argmax','')}"
        )
    return "\n".join(lines)


def prev_trading_days(sym: str, date_iso: str, n: int) -> list[str]:
    month = date_iso[:7]
    candidates = [
        OFFLINE_ROOT / sym / "regular/09:30-16:00/1min" / f"{month}.parquet",
    ]
    fb = PITCHER_FALLBACK_ROOTS.get(sym)
    if fb is not None:
        candidates.append(fb / f"{month}.parquet")
    for path in candidates:
        if not path.exists():
            continue
        ts = _ny(pd.read_parquet(path, columns=["timestamp"])["timestamp"])
        all_days = sorted(set(ts.dt.date))
        target = pd.Timestamp(date_iso).date()
        days = [str(d) for d in all_days if d < target][-n:]
        if days:
            return days
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description="honest sim 秒级对拍(无注入)")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--warmup-days", type=int, default=1, help="price 特征历史窗口前置天数")
    ap.add_argument("--price-tol", type=float, default=1e-3)
    ap.add_argument("--ohlc-tol", type=float, default=1e-6)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    date_iso = args.date
    sym = args.symbol
    report: dict = {"date": date_iso, "symbol": sym}

    # ---------- L0: 发球机源 1m vs 离线 raw ----------
    offline = load_offline_day(sym, date_iso)
    pitcher_1m, src_tag = load_pitcher_source_1m(sym, date_iso)
    print(f"=== L0 发球机数据源 1m vs quote_features_raw | source={src_tag} ===")
    if offline.empty or pitcher_1m.empty:
        print(f"  offline rows={len(offline)} pitcher rows={len(pitcher_1m)} → 无法对比")
        rows0 = []
    else:
        cmp_src = pitcher_1m.copy()
        # 离线终表 vwap 列是"日内累计"口径;数据源 vwap 是"分钟"口径 → 用同口径累计值对比
        vol = pd.to_numeric(cmp_src["volume"], errors="coerce").clip(lower=0)
        cmp_src["vwap"] = (cmp_src["close"] * vol).cumsum() / (vol.cumsum() + 1e-9)
        # 离线终表 volume 被截成 int64,数据源是 float → 同样截断后对比
        cmp_src["volume"] = np.floor(vol)
        rows0 = diff_frame(offline, cmp_src, ["open", "high", "low", "close", "volume", "vwap"])
        print(fmt_rows(rows0, args.ohlc_tol))
    report["L0_source_vs_offline_ohlc"] = {"source": src_tag, "rows": rows0}

    # ---------- L2: FCS pandas 特征 (用发球机源 1m + warmup) vs 离线列 ----------
    print(f"\n=== L2 FCS pandas 特征 (发球机源 OHLC, warmup={args.warmup_days}d) vs 离线列 ===")
    hist_parts = []
    for d in prev_trading_days(sym, date_iso, args.warmup_days):
        prev_1m, _ = load_pitcher_source_1m(sym, d)
        if not prev_1m.empty:
            hist_parts.append(prev_1m)
    hist_parts.append(pitcher_1m)
    hist_all = pd.concat(hist_parts).sort_index() if hist_parts else pd.DataFrame()

    rows2 = []
    if hist_all.empty or offline.empty:
        print("  数据不足,跳过")
    else:
        feats = fcs_price_features(hist_all, PRICE_FEATURES)
        feats_day = feats[feats.index.date == pd.Timestamp(date_iso).date()]
        rows2 = diff_frame(offline, feats_day, list(PRICE_FEATURES))
        print(fmt_rows(rows2, args.price_tol))
    report["L2_fcs_price_features"] = rows2

    # ---------- L3: vix_level (VIXY) cold vs warm ----------
    print("\n=== L3 vix_level: FCS 公式 on 发球机 VIXY 源 vs 离线列 (cold vs warm) ===")
    vixy_offline = load_offline_day("VIXY", date_iso)
    vixy_pitcher, vixy_tag = load_pitcher_source_1m("VIXY", date_iso)
    rows3 = []
    if vixy_offline.empty and "vix_level" not in offline.columns:
        print("  离线无 vix_level 参照,跳过")
    elif vixy_pitcher.empty:
        print(f"  发球机 VIXY 源为空({vixy_tag}),跳过")
    else:
        # 离线参照优先 QQQ 表中 merge 好的 vix_level 列
        ref = offline[["vix_level"]] if "vix_level" in offline.columns else vixy_offline[["vix_level"]]
        print(f"  source={vixy_tag}")
        target_d = pd.Timestamp(date_iso).date()

        # warm: 前 warmup_days 交易日 VIXY 历史 + 当日,截取当日 rows(模拟 FCS warm 预热)
        warm_parts = []
        for d in prev_trading_days("VIXY", date_iso, max(1, args.warmup_days)):
            prev_v, _ = load_pitcher_source_1m("VIXY", d)
            if not prev_v.empty:
                warm_parts.append(prev_v)
        warm_parts.append(vixy_pitcher)
        vixy_warm = pd.concat(warm_parts).sort_index()

        for label, series in (
            ("cold(仅当日)", fcs_vix_level_series(vixy_pitcher["close"])),
            (
                f"warm(+{max(1, args.warmup_days)}d 前史)",
                fcs_vix_level_series(vixy_warm["close"]).loc[
                    lambda s: s.index.date == target_d
                ],
            ),
        ):
            online_vl = series.to_frame("vix_level")
            r = diff_frame(ref, online_vl, ["vix_level"])
            for item in r:
                item["col"] = f"vix_level[{label}]"
            print(fmt_rows(r, 0.25))
            idx = ref.index.intersection(online_vl.index)
            if len(idx) > 30:
                corr = float(
                    pd.to_numeric(ref.loc[idx, "vix_level"], errors="coerce").corr(
                        online_vl.loc[idx, "vix_level"]
                    )
                )
                print(f"    corr[{label}] = {corr:.3f}")
                r.append({"col": f"vix_level_corr[{label}]", "n": len(idx), "med": corr, "max": corr})
            rows3.extend(r)
    report["L3_vix_level"] = {"source": vixy_tag, "rows": rows3}

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        print(f"\nWrote {args.json_out}")

    gaps = [r["col"] for r in rows0 + rows2 if r.get("med") is not np.nan and not r.get("note") and r["med"] > args.price_tol]
    print(f"\n=== verdict === gaps: {gaps if gaps else '无(L0/L2 在容差内)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
