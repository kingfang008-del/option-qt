#!/usr/bin/env python3
"""
rails_value LightGBM 验证 v2:特征扩容版。

v1(rails_value_lgbm.py)结论:仅用主腿报价衍生特征,IC 卡在盈亏平衡线
(0.07~0.13 vs 门槛 0.10~0.12),瓶颈在特征信息量。

v2 新增三类信息源:
  1. 真实现货 1s OHLCV(/mnt/s990/data/raw_1s/stocks/QQQ):
     多尺度动量、已实现波动、量能 z、上/下行成交量不平衡、VWAP 距离
  2. 跨 bucket 期权结构(同文件 bucket 0/1/3/4/5):
     PUT/CALL 偏度(OTM/ATM 价比)、期限结构(短/长straddle比)及其变化
  3. 报价强度:主腿每分钟 tick 数、分钟内 mid 波动、盘口 size 动态

协议与 v1 相同:train 2025-01~04 / val 05 / test 06(另跑 05 交叉),
标签 binary top-10%(v1 验证的最优形式),因果 top-k replay。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qqq_btc.common.event_replay import EventReplayConfig, run_event_replay
from qqq_btc.qqq import config as qcfg

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from raw1s_rule_validation import (  # noqa: E402
    build_minute_frame,
    compute_oracle_edge,
    discover_raw1s_days,
)
from rails_value_lgbm import (  # noqa: E402
    ENTRY_END,
    ENTRY_START,
    _r2_of,
    _round4,
    _slope_of,
    causal_topk_signal,
    compute_rails_value,
    daily_rank_ic,
    entry_mask,
)

SPOT_DIR = Path("/mnt/s990/data/raw_1s/stocks/QQQ")

FEATURES_V2: List[str] = []


# ---------------------------------------------------------------------------
# 数据装载:每日 parquet 只读一次,拆出全部 bucket
# ---------------------------------------------------------------------------
def _normalize_option_ticks(sub: pd.DataFrame) -> pd.DataFrame:
    """与 load_raw1s_bucket_day 相同的规范化(bid/ask/mid/spread + RTH)。"""
    if sub.empty:
        return pd.DataFrame()
    sub = sub.copy()
    if not pd.api.types.is_datetime64_any_dtype(sub["timestamp"]):
        sub["timestamp"] = pd.to_datetime(sub["timestamp"])
    if sub["timestamp"].dt.tz is None:
        sub["timestamp"] = sub["timestamp"].dt.tz_localize(
            "America/New_York", ambiguous="infer"
        )
    else:
        sub["timestamp"] = sub["timestamp"].dt.tz_convert("America/New_York")
    sub = sub.sort_values("timestamp")
    bid = pd.to_numeric(sub["bid"], errors="coerce")
    ask = pd.to_numeric(sub["ask"], errors="coerce")
    mid = (bid + ask) / 2.0
    sub = sub.assign(
        exec_call_bid=bid,
        exec_call_ask=ask,
        exec_call_mid=mid,
        exec_call_spread_pct=np.where(mid > 0, (ask - bid) / mid, np.nan),
        bid_size=pd.to_numeric(sub.get("bid_size", 0), errors="coerce").fillna(0),
        ask_size=pd.to_numeric(sub.get("ask_size", 0), errors="coerce").fillna(0),
        minute_ts=sub["timestamp"].dt.floor("min"),
    )
    t = sub["timestamp"]
    rth = ((t.dt.hour > 9) | ((t.dt.hour == 9) & (t.dt.minute >= 30))) & (t.dt.hour < 16)
    return sub[rth].reset_index(drop=True)


def load_day_buckets(fp: Path) -> Dict[int, pd.DataFrame]:
    df = pd.read_parquet(fp)
    df["bucket_id"] = pd.to_numeric(df["bucket_id"], errors="coerce")
    out = {}
    for b, grp in df.groupby("bucket_id"):
        out[int(b)] = _normalize_option_ticks(grp)
    return out


def leg_minute(ticks: pd.DataFrame, prefix: str) -> Optional[pd.DataFrame]:
    """腿的分钟 mid/spread(取每分钟最后一笔)。"""
    if ticks is None or ticks.empty:
        return None
    g = ticks.groupby("minute_ts").agg(
        mid=("exec_call_mid", "last"), spread=("exec_call_spread_pct", "last")
    )
    g.columns = [f"{prefix}_mid", f"{prefix}_spread"]
    return g.reset_index().rename(columns={"minute_ts": "timestamp"})


def quote_intensity_minute(ticks: pd.DataFrame) -> pd.DataFrame:
    """主腿报价强度:每分钟 tick 数 + 分钟内 mid 收益 std + size 均值。"""
    g = ticks.groupby("minute_ts").agg(
        q_ticks=("exec_call_mid", "size"),
        q_intramin_std=("exec_call_mid", lambda s: float(np.nanstd(np.diff(np.log(s[s > 0])))) if (s > 0).sum() > 2 else np.nan),
        q_bidsz=("bid_size", "mean"),
        q_asksz=("ask_size", "mean"),
    )
    return g.reset_index().rename(columns={"minute_ts": "timestamp"})


def load_spot_minute(date_str: str) -> Optional[pd.DataFrame]:
    fp = SPOT_DIR / f"QQQ_{date_str}.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("America/New_York", ambiguous="infer")
    else:
        ts = ts.dt.tz_convert("America/New_York")
    df = df.assign(timestamp=ts)
    rth = ((ts.dt.hour > 9) | ((ts.dt.hour == 9) & (ts.dt.minute >= 30))) & (ts.dt.hour < 16)
    df = df[rth].sort_values("timestamp")
    if df.empty:
        return None
    df["minute_ts"] = df["timestamp"].dt.floor("min")
    ret1s = df["close"].pct_change()
    df["up_vol"] = np.where(ret1s > 0, df["volume"], 0.0)
    df["dn_vol"] = np.where(ret1s < 0, df["volume"], 0.0)
    g = df.groupby("minute_ts").agg(
        s_open=("open", "first"),
        s_high=("high", "max"),
        s_low=("low", "min"),
        s_close=("close", "last"),
        s_volume=("volume", "sum"),
        s_upvol=("up_vol", "sum"),
        s_dnvol=("dn_vol", "sum"),
        s_rv1s=("close", lambda s: float(np.nanstd(np.diff(np.log(s[s > 0])))) if (s > 0).sum() > 2 else np.nan),
    )
    return g.reset_index().rename(columns={"minute_ts": "timestamp"})


# ---------------------------------------------------------------------------
# v2 特征
# ---------------------------------------------------------------------------
def build_features_v2(minute: pd.DataFrame) -> pd.DataFrame:
    m = minute.copy()
    mid = m["exec_call_mid"]
    logm = np.log(mid.where(mid > 0))
    feats: Dict[str, pd.Series] = {}

    # --- 主腿(与 v1 相同的核心子集) ---
    for w in (1, 3, 5, 15, 30, 60):
        feats[f"ret_{w}m"] = logm.diff(w)
    for w in (10, 30, 60):
        feats[f"vol_{w}m"] = logm.diff().rolling(w, min_periods=max(3, w // 3)).std()
    feats["vol_ratio_10_60"] = feats["vol_10m"] / feats["vol_60m"]
    roll_max30 = mid.rolling(30, min_periods=5).max()
    roll_min30 = mid.rolling(30, min_periods=5).min()
    feats["dd_from_max30"] = mid / roll_max30 - 1.0
    feats["range_pos_30"] = (mid - roll_min30) / (roll_max30 - roll_min30)
    sp = m["exec_call_spread_pct"]
    feats["spread_pct"] = sp
    feats["spread_z30"] = (sp - sp.rolling(30, min_periods=5).mean()) / (
        sp.rolling(30, min_periods=5).std() + 1e-9
    )
    bidsz = m["bid_size"].astype(float)
    asksz = m["ask_size"].astype(float)
    feats["size_imb"] = (bidsz - asksz) / (bidsz + asksz + 1e-9)
    feats["size_imb_ma10"] = feats["size_imb"].rolling(10, min_periods=3).mean()
    sb = m["session_bar"].astype(float)
    feats["session_bar"] = sb
    feats["tod_sin"] = np.sin(2 * np.pi * sb / 390.0)
    feats["tod_cos"] = np.cos(2 * np.pi * sb / 390.0)
    feats["trend_slope_30"] = logm.rolling(30).apply(_slope_of, raw=True)
    feats["trend_r2_30"] = logm.rolling(30).apply(_r2_of, raw=True)

    # --- 真实现货 ---
    if "s_close" in m.columns:
        sc = m["s_close"].ffill()
        logs = np.log(sc.where(sc > 0))
        for w in (1, 5, 15, 30, 60):
            feats[f"s_ret_{w}m"] = logs.diff(w)
        feats["s_vol_30m"] = logs.diff().rolling(30, min_periods=10).std()
        feats["s_rv1s"] = m["s_rv1s"]
        feats["s_rv1s_z30"] = (m["s_rv1s"] - m["s_rv1s"].rolling(30, min_periods=5).mean()) / (
            m["s_rv1s"].rolling(30, min_periods=5).std() + 1e-12
        )
        smax30 = sc.rolling(30, min_periods=5).max()
        smin30 = sc.rolling(30, min_periods=5).min()
        feats["s_range_pos_30"] = (sc - smin30) / (smax30 - smin30)
        feats["s_trend_slope_30"] = logs.rolling(30).apply(_slope_of, raw=True)
        feats["s_trend_r2_30"] = logs.rolling(30).apply(_r2_of, raw=True)
        # 量能 / 订单流
        vol = m["s_volume"].astype(float)
        feats["s_volume_z30"] = (vol - vol.rolling(30, min_periods=5).mean()) / (
            vol.rolling(30, min_periods=5).std() + 1e-9
        )
        flow = (m["s_upvol"] - m["s_dnvol"]) / (m["s_upvol"] + m["s_dnvol"] + 1e-9)
        feats["s_flow_imb"] = flow
        feats["s_flow_imb_ma10"] = flow.rolling(10, min_periods=3).mean()
        feats["s_flow_imb_sum30"] = flow.rolling(30, min_periods=10).sum()
        cum_pv = (sc * vol).cumsum()
        cum_v = vol.cumsum()
        vwap = cum_pv / (cum_v + 1e-9)
        feats["s_vwap_dist"] = sc / vwap - 1.0
        hl = (m["s_high"].ffill() - m["s_low"].ffill()) / sc
        feats["s_hl_range"] = hl
        feats["s_hl_range_ma15"] = hl.rolling(15, min_periods=5).mean()
        feats["s_close_pos_hl"] = (sc - m["s_low"].ffill()) / (
            m["s_high"].ffill() - m["s_low"].ffill() + 1e-9
        )

    # --- 跨 bucket 期权结构 ---
    def _safe_log_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
        return np.log((a / b).where((a > 0) & (b > 0)))

    if "put_atm_mid" in m.columns:
        pa = m["put_atm_mid"].ffill()
        feats["cp_ratio"] = _safe_log_ratio(mid, pa)
        feats["cp_ratio_chg15"] = feats["cp_ratio"].diff(15)
        straddle = mid + pa
        feats["straddle_ret_15m"] = np.log(straddle.where(straddle > 0)).diff(15)
        feats["put_atm_spread"] = m["put_atm_spread"]
    if "put_otm_mid" in m.columns and "put_atm_mid" in m.columns:
        skew_p = _safe_log_ratio(m["put_otm_mid"].ffill(), m["put_atm_mid"].ffill())
        feats["skew_put"] = skew_p
        feats["skew_put_chg15"] = skew_p.diff(15)
    if "call_otm_mid" in m.columns:
        skew_c = _safe_log_ratio(m["call_otm_mid"].ffill(), mid)
        feats["skew_call"] = skew_c
        feats["skew_call_chg15"] = skew_c.diff(15)
    if "put_long_mid" in m.columns and "call_long_mid" in m.columns and "put_atm_mid" in m.columns:
        straddle_s = mid + m["put_atm_mid"].ffill()
        straddle_l = m["call_long_mid"].ffill() + m["put_long_mid"].ffill()
        term = _safe_log_ratio(straddle_s, straddle_l)
        feats["term_ratio"] = term
        feats["term_ratio_chg15"] = term.diff(15)
        feats["long_straddle_ret15"] = np.log(straddle_l.where(straddle_l > 0)).diff(15)

    # --- 报价强度 ---
    if "q_ticks" in m.columns:
        qt = m["q_ticks"].astype(float)
        feats["q_ticks_z30"] = (qt - qt.rolling(30, min_periods=5).mean()) / (
            qt.rolling(30, min_periods=5).std() + 1e-9
        )
        feats["q_intramin_std"] = m["q_intramin_std"]
        feats["q_intramin_std_z30"] = (
            m["q_intramin_std"] - m["q_intramin_std"].rolling(30, min_periods=5).mean()
        ) / (m["q_intramin_std"].rolling(30, min_periods=5).std() + 1e-12)

    for k, v in feats.items():
        m[k] = v
    global FEATURES_V2
    FEATURES_V2 = sorted(feats.keys())
    return m


# ---------------------------------------------------------------------------
# 装载全流程
# ---------------------------------------------------------------------------
LEG_MAP = {0: "put_atm", 1: "put_otm", 3: "call_otm", 4: "put_long", 5: "call_long"}


def load_month_days(
    raw_dir: Path, symbol: str, globs: Sequence[str], bucket: int
) -> List[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    out = []
    for g in globs:
        for fp in discover_raw1s_days(raw_dir, symbol, glob_pattern=g):
            date_str = fp.stem.split("_", 1)[-1]
            buckets = load_day_buckets(fp)
            ticks = buckets.get(bucket)
            if ticks is None or ticks.empty:
                continue
            minute = build_minute_frame(ticks)
            if minute.empty or len(minute) < 120:
                continue
            minute = compute_rails_value(minute, qcfg.FILL_MODEL, qcfg.EXIT_RAILS)
            minute = compute_oracle_edge(minute, qcfg.FILL_MODEL, hold_bars=5)

            for b, prefix in LEG_MAP.items():
                lm = leg_minute(buckets.get(b), prefix)
                if lm is not None:
                    minute = minute.merge(lm, on="timestamp", how="left")
            minute = minute.merge(quote_intensity_minute(ticks), on="timestamp", how="left")
            spot = load_spot_minute(date_str)
            if spot is not None:
                minute = minute.merge(spot, on="timestamp", how="left")

            minute = build_features_v2(minute)
            tick_df = ticks[
                ["timestamp", "exec_call_bid", "exec_call_ask", "exec_call_spread_pct"]
            ]
            out.append((date_str, minute, tick_df))
    return out


def to_xy(days, label_top: Optional[float] = None):
    Xs, ys, ds = [], [], []
    for di, (_date, minute, _t) in enumerate(days):
        w = entry_mask(minute) & minute["rails_value"].notna()
        sub = minute.loc[w]
        y = sub["rails_value"]
        if label_top is not None:
            y = (y.rank(pct=True) >= 1.0 - label_top).astype(float)
        Xs.append(sub[FEATURES_V2].to_numpy(dtype=np.float64))
        ys.append(y.to_numpy(dtype=np.float64))
        ds.append(np.full(len(sub), di))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ds)


def replay_month(days, preds_by_day, top_pct: float, momentum_gate: Optional[float] = None) -> dict:
    day_rois, hits = [], []
    total_trades = 0
    exit_counts: Dict[str, int] = {}
    for date_str, minute, tick_df in days:
        pred = preds_by_day[date_str]
        w = entry_mask(minute).to_numpy()
        gate = None
        if momentum_gate is not None:
            gate = minute["ret_15m"].to_numpy() > momentum_gate
        m = minute.copy()
        m["lgbm_signal"] = causal_topk_signal(pred, w, top_pct, gate=gate)
        r = run_event_replay(
            m,
            qcfg.FILL_MODEL,
            qcfg.REPLAY,
            qcfg.EXIT_RAILS,
            tick_df=tick_df,
            edge_col="lgbm_signal",
            event_cfg=EventReplayConfig(tick_disaster_stop=True),
        )
        if not r.trades:
            day_rois.append(0.0)
            continue
        rets = np.array([t.net_return for t in r.trades])
        day_rois.append(float(np.prod(1.0 + rets) - 1.0))
        total_trades += len(rets)
        hits.extend((rets > 0).astype(float).tolist())
        for t in r.trades:
            exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1
    dr = np.array(day_rois)
    return {
        "top_pct": top_pct,
        "momentum_gate": momentum_gate,
        "days": len(dr),
        "active_days": int((dr != 0).sum()),
        "win_days": int((dr > 0).sum()),
        "trades": total_trades,
        "hit_rate": _round4(float(np.mean(hits))) if hits else 0.0,
        "day_roi_mean": _round4(float(dr.mean())),
        "compound": _round4(float(np.prod(1.0 + dr) - 1.0)),
        "worst_day": _round4(float(dr.min())) if len(dr) else 0.0,
        "exit_reasons": exit_counts,
    }


def selection_diag(days, preds_by_day, top_pct: float = 0.02) -> dict:
    sel_vals, hit_top10 = [], []
    for date_str, minute, _t in days:
        pred = preds_by_day[date_str]
        w = entry_mask(minute).to_numpy()
        sig = causal_topk_signal(pred, w, top_pct)
        rvv = minute["rails_value"].to_numpy()
        sel = (sig > 0) & np.isfinite(rvv)
        ok = w & np.isfinite(rvv)
        if not sel.any() or not ok.any():
            continue
        thr = np.nanquantile(rvv[ok], 0.90)
        sel_vals.extend(rvv[sel].tolist())
        hit_top10.extend((rvv[sel] >= thr).astype(float).tolist())
    return {
        "sel_rails_mean": _round4(float(np.mean(sel_vals))) if sel_vals else None,
        "sel_oracle_top10_hit": _round4(float(np.mean(hit_top10))) if hit_top10 else None,
        "n_selected": len(sel_vals),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="rails-value LGBM v2 (expanded features)")
    ap.add_argument("--raw-1s-dir", default="/mnt/s990/data/raw_1s/dte1_options")
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--bucket", type=int, default=2)
    ap.add_argument("--train-globs", default="QQQ_2025-01-*.parquet,QQQ_2025-02-*.parquet,QQQ_2025-03-*.parquet,QQQ_2025-04-*.parquet")
    ap.add_argument("--val-globs", default="QQQ_2025-05-*.parquet")
    ap.add_argument(
        "--test-globs",
        default="QQQ_2025-06-*.parquet",
        help="多个独立测试段用 ';' 分隔,每段内部可用 ',' 组合多个 glob",
    )
    ap.add_argument("--label-top", type=float, default=0.10)
    ap.add_argument("--top-pcts", default="0.02,0.05")
    ap.add_argument(
        "--out",
        default="New_Pro/baseline_qqq/reports/qqq_1dte_rails_value_lgbm_v2.json",
    )
    args = ap.parse_args()

    import lightgbm as lgb

    raw_dir = Path(args.raw_1s_dir).expanduser()
    top_pcts = [float(x) for x in args.top_pcts.split(",")]

    print("loading train ...")
    train_days = load_month_days(raw_dir, args.symbol, args.train_globs.split(","), args.bucket)
    print(f"  train days={len(train_days)}")
    print("loading val ...")
    val_days = load_month_days(raw_dir, args.symbol, args.val_globs.split(","), args.bucket)
    print(f"  val days={len(val_days)}")
    test_segments: List[Tuple[str, list]] = []
    for seg in args.test_globs.split(";"):
        print(f"loading test [{seg}] ...")
        seg_days = load_month_days(raw_dir, args.symbol, seg.split(","), args.bucket)
        print(f"  test days={len(seg_days)}")
        test_segments.append((seg, seg_days))

    Xtr, ytr, dtr = to_xy(train_days, label_top=args.label_top)
    Xva, yva, dva = to_xy(val_days, label_top=args.label_top)
    print(f"rows train={len(ytr)} val={len(yva)} feats={len(FEATURES_V2)}")

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=2.0,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    ic_tr = daily_rank_ic(model.predict(Xtr), ytr, dtr)
    ic_va = daily_rank_ic(model.predict(Xva), yva, dva)
    print(f"rank IC  train={np.mean(ic_tr):+.3f}  val={np.mean(ic_va):+.3f}")

    imp = sorted(zip(FEATURES_V2, model.feature_importances_), key=lambda x: -x[1])
    print("top features:", [k for k, _v in imp[:12]])

    segments_out = []
    for seg_name, test_days in test_segments:
        Xte, yte, dte = to_xy(test_days)  # test 对 raw rails_value 测 IC
        ic_te = daily_rank_ic(model.predict(Xte), yte, dte)
        print(
            f"\n[{seg_name}] rank IC test={np.mean(ic_te):+.3f} "
            f"(day IC>0: {np.mean(np.array(ic_te) > 0):.0%})"
        )

        preds_by_day = {}
        for date_str, minute, _t in test_days:
            preds_by_day[date_str] = model.predict(
                minute[FEATURES_V2].to_numpy(dtype=np.float64)
            )
        diag = selection_diag(test_days, preds_by_day)
        print(f"[{seg_name}] selection diag (top2%): {diag}")

        replay_rows = []
        for pct in top_pcts:
            for mg in (None, 0.0):
                rr = replay_month(test_days, preds_by_day, pct, momentum_gate=mg)
                replay_rows.append(rr)
                tag = "no_gate" if mg is None else f"mom>{mg:+.2f}"
                print(
                    f"[{seg_name}] replay top{pct:.0%} [{tag}]: win={rr['win_days']}/{rr['days']} "
                    f"dayROI={rr['day_roi_mean']:+.1%} comp={rr['compound']:+.1%} "
                    f"trades={rr['trades']} hit={rr['hit_rate']:.0%} worst_day={rr['worst_day']:+.1%}"
                )
        segments_out.append({
            "segment": seg_name,
            "rank_ic_test_mean": _round4(float(np.mean(ic_te))),
            "test_daily_ic": [_round4(v) for v in ic_te],
            "test_pos_day_frac": _round4(float(np.mean(np.array(ic_te) > 0))),
            "selection_diag_top2pct": diag,
            "test_replay": replay_rows,
        })

    result = {
        "meta": {
            "version": "v2_expanded_features",
            "train_globs": args.train_globs,
            "val_globs": args.val_globs,
            "test_globs": args.test_globs,
            "label_top": args.label_top,
            "n_features": len(FEATURES_V2),
            "features": FEATURES_V2,
        },
        "rank_ic": {
            "train_mean": _round4(float(np.mean(ic_tr))),
            "val_mean": _round4(float(np.mean(ic_va))),
        },
        "feature_importance_top20": [[k, int(v)] for k, v in imp[:20]],
        "test_segments": segments_out,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
