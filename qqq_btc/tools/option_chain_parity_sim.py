#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线仿真验证：修复后的 FCS 期权分钟链路 vs 离线 quote_features_raw。

模拟修复后的因果链（无 Redis，秒级完成）：
  raw_1s cbbo quotes → ceil 分钟槽 (label M = 状态 asof M)
  → 分钟成交量 ref[M] 注入 (quote_options_day_iv, 数据源补充, 非特征注入)
  → BSM 反解 IV (mid, T 锚 16:00 到期, spot=quote 同时刻正股价)
  → options_locked_feature 公式 → 与离线 1min 特征逐分钟对比

用法:
  python qqq_btc/tools/option_chain_parity_sim.py --date 2026-06-15 [--symbol QQQ]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

NY = "America/New_York"
OPTION_ROOT = Path("/mnt/s990/data/raw_1s/options_databento_v3")
GREEK_ROOT = Path.home() / "train_data/quote_options_day_iv"
OFFLINE_ROOT = Path.home() / "train_data/quote_features_raw"
STOCK_1M_ROOT = Path.home() / "train_data/spnq_train/QQQ"

EPS = 1e-9


def load_offline_day(sym: str, date_iso: str, offline_root: Path = OFFLINE_ROOT) -> pd.DataFrame:
    month = date_iso[:7]
    root = Path(offline_root).expanduser()
    candidates = [
        root / sym / "regular/09:30-16:00/1min" / f"{month}.parquet",
        root / "regular/09:30-16:00/1min" / f"{month}.parquet",
        root / f"{month}.parquet",
    ]
    fp = next((p for p in candidates if p.exists()), candidates[0])
    df = pd.read_parquet(fp)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
    df = df[df["timestamp"].dt.date == pd.Timestamp(date_iso).date()]
    return df.set_index("timestamp").sort_index()


def load_stock_minutes(date_iso: str) -> pd.DataFrame:
    month = date_iso[:7]
    fp = STOCK_1M_ROOT / f"{month}.parquet"
    if not fp.exists():
        fp = STOCK_1M_ROOT / "regular/09:30-16:00/1min" / f"{month}.parquet"
    df = pd.read_parquet(fp)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
    df = df[df["timestamp"].dt.date == pd.Timestamp(date_iso).date()]
    return df.set_index("timestamp")[["open", "close"]].sort_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-06-15")
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--option-root", default=str(OPTION_ROOT))
    ap.add_argument("--offline-root", default=str(OFFLINE_ROOT))
    ap.add_argument(
        "--minute-ref-state",
        action="store_true",
        help="诊断上界：用 monthly minute_ref 的 spread_pct/volume_imbalance 重建盘口状态",
    )
    args = ap.parse_args()
    sym, date_iso = args.symbol, args.date

    from py_vollib_vectorized import vectorized_implied_volatility, vectorized_delta

    from qqq_btc.common.option_minute_ref import load_minute_option_ref

    raw = pd.read_parquet(Path(args.option_root).expanduser() / sym / f"{sym}_{date_iso}.parquet")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(NY)
    raw = raw.sort_values("timestamp")

    ref = load_minute_option_ref(sym, date_iso, greek_root=GREEK_ROOT)
    off = load_offline_day(sym, date_iso, Path(args.offline_root))
    stock_close = load_stock_minutes(date_iso)

    minutes = pd.date_range(
        f"{date_iso} 09:31", f"{date_iso} 16:00", freq="1min", tz=NY
    )
    expiry = None

    # 每桶 quote 状态 asof M (ceil 槽语义: 发球机 1s ffill → 槽 M 的最后 tick 即 asof M)
    per_bucket: dict[int, pd.DataFrame] = {}
    for b_id, grp in raw.groupby("bucket_id"):
        g = grp.drop_duplicates("timestamp", keep="last").set_index("timestamp")
        per_bucket[int(b_id)] = g

    rows = []
    prev_snap = None
    for m in minutes:
        snap = np.zeros((6, 12), dtype=np.float64)
        types = [""] * 6
        for b_id, g in per_bucket.items():
            if not (0 <= b_id <= 5):
                continue
            sel = g[g.index <= m]
            if sel.empty:
                continue
            r = sel.iloc[-1]
            bid, ask = float(r["bid"]), float(r["ask"])
            mid = (bid + ask) / 2.0 if (bid > 0 and ask >= bid) else 0.0
            snap[b_id, 0] = mid
            snap[b_id, 5] = float(r["strike"])
            snap[b_id, 8], snap[b_id, 9] = bid, ask
            snap[b_id, 10] = float(r.get("bid_size", 0.0) or 0.0)
            snap[b_id, 11] = float(r.get("ask_size", 0.0) or 0.0)
            tkr_s = str(r["ticker"]).replace("O:", "")
            types[b_id] = "p" if "P" in tkr_s[len(sym) + 6: len(sym) + 7] else "c"
            if expiry is None:
                tkr = str(r["ticker"])
                ymd = tkr.replace("O:", "")[len(sym): len(sym) + 6]
                expiry = pd.to_datetime(ymd, format="%y%m%d").tz_localize(NY) + pd.Timedelta(hours=16)
        # ref 量注入 (ceil key = M); ref 缺该分钟 → volume 0 (对齐离线 ffill 语义)
        mts = int(m.timestamp())
        for b_id in range(4):
            info = ref.get((mts, b_id))
            snap[b_id, 6] = float(info["volume"]) if info else 0.0
            if args.minute_ref_state and info:
                mid = float(snap[b_id, 0])
                sp = float(info.get("spread_pct", 0.0) or 0.0)
                if mid > 1e-6 and sp >= 0.0:
                    half = mid * sp * 0.5
                    snap[b_id, 8] = max(mid - half, 1e-6)
                    snap[b_id, 9] = max(mid + half, snap[b_id, 8])
                imb = info.get("volume_imbalance")
                if imb is not None and np.isfinite(float(imb)):
                    imb_f = float(np.clip(imb, -0.99, 0.99))
                    base = 1000.0
                    snap[b_id, 10] = base * (1.0 + imb_f)
                    snap[b_id, 11] = base * (1.0 - imb_f)

        # 离线 dayIV 的 stock_close(M) 实测 = bar M 收盘价 (分钟末)。
        # FCS commit 分钟 M 时 committed bar M 已就绪，engine 用 last_row['close'] 天然一致。
        prior = stock_close[stock_close.index <= m]
        spot = float(prior["close"].iloc[-1]) if not prior.empty else 0.0

        # BSM 反解 (engine supplement_greeks 口径: mid, ts_anchor=M, expiry 16:00, r=rfr)
        T = max((expiry - m).total_seconds() / 31557600.0, 1e-6)
        for b_id in range(4):
            mid = snap[b_id, 0]
            k = snap[b_id, 5]
            if mid > 1e-4 and k > 0.01 and spot > 0.01 and types[b_id]:
                iv = vectorized_implied_volatility(
                    np.array([mid]), np.array([spot]), np.array([k]),
                    np.array([T]), np.array([0.04]), types[b_id],
                    return_as="numpy", on_error="ignore",
                )[0]
                iv = float(iv) if np.isfinite(iv) else 0.0
                snap[b_id, 7] = iv
                if iv > 0:
                    snap[b_id, 1] = float(
                        vectorized_delta(
                            types[b_id], np.array([spot]), np.array([k]),
                            np.array([T]), np.array([0.04]), np.array([iv]),
                            return_as="numpy",
                        )[0]
                    )

        # options_locked_feature 公式 (与 engine._calc_opt_feats_batch 对齐)
        v = snap[:, 6]
        iv = snap[:, 7]
        total = v[:4].sum()
        no_vol = total < 1.0
        vw_iv = (iv[0] + iv[2]) / 2.0
        spread_pct = (snap[:, 9] - snap[:, 8]) / (snap[:, 0] + EPS)
        imb = (snap[:4, 10] - snap[:4, 11]) / (snap[:4, 10] + snap[:4, 11] + EPS)
        vw_spread = 0.0 if no_vol else float((spread_pct[:4] * v[:4]).sum() / (total + EPS))
        vw_imb = 0.0 if no_vol else float((imb * v[:4]).sum() / (total + EPS))
        vw_delta = 0.0 if no_vol else float((snap[:4, 1] * v[:4]).sum() / (total + EPS))
        pcr = (v[0] + v[1]) / (v[2] + v[3]) if (v[2] + v[3]) > 0 else 1.0
        struc_skew = iv[1] / (iv[0] + EPS) if iv[0] > 0.01 else 1.0
        flow_skew = iv[1] / (iv[3] + EPS) if iv[3] > 0.01 else 1.0
        rows.append({
            "timestamp": m,
            "options_vw_iv": vw_iv,
            "options_struc_atm_iv": vw_iv,
            "options_vw_spread": vw_spread,
            "options_vw_imbalance": vw_imb,
            "options_vw_delta": vw_delta,
            "options_pcr_volume": pcr,
            "options_struc_skew": struc_skew,
            "options_flow_skew": flow_skew,
        })
        prev_snap = snap

    sim = pd.DataFrame(rows).set_index("timestamp")
    sim["options_iv_momentum"] = sim["options_vw_iv"].pct_change(5).fillna(0.0)

    feats = [
        "options_vw_iv", "options_struc_atm_iv", "options_vw_spread",
        "options_vw_imbalance", "options_vw_delta", "options_pcr_volume",
        "options_struc_skew", "options_flow_skew", "options_iv_momentum",
    ]
    idx = sim.index.intersection(off.index)
    print(f"{'feature':26s} {'corr':>7s} {'med_abs':>10s} {'max_abs':>10s}")
    for f in feats:
        if f not in off.columns:
            continue
        a = pd.to_numeric(off.loc[idx, f], errors="coerce")
        b = sim.loc[idx, f]
        m2 = a.notna() & b.notna()
        d = (a[m2] - b[m2]).abs()
        corr = a[m2].corr(b[m2])
        print(f"{f:26s} {corr:7.3f} {d.median():10.4g} {d.max():10.4g}")


if __name__ == "__main__":
    main()
