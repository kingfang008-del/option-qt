#!/usr/bin/env python3
"""
反事实 rails 价值标签 + LightGBM 因果特征验证。

标签(rails_value):对每个入场窗 bar i,假设在 bar i+1 以 fill 价入场,
然后由生产 EXIT_RAILS(含 vol-scale、EOD 强平)逐分钟推进直至退出,
记录净收益。这就是"如果模型在这根 bar 发出信号,系统实际会赚/亏多少",
与训练目标和真实 PnL 完全对齐(替代固定 horizon 收益标签)。

验证(LightGBM):只用因果特征(过去的期权 mid/spread/量),训练回归
rails_value,在时间外推月上测:
  1) 逐日 rank IC(预测 vs rails_value)
  2) 因果 top-k 入场 replay:把预测转成"日内截至当前的历史分位",
     分位 >= 1-k 才给入场资格,跑与生产一致的 replay 栈。
对照锚点(2025-06):oracle 连续 +3806%,oracle 仅排序 top5% +142179%,
纯噪声 -50%(见 oracle_noise_ablation)。

输出 JSON 报告。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qqq_btc.common.event_replay import EventReplayConfig, run_event_replay
from qqq_btc.common.exit_rails import (
    ExitRailsConfig,
    PositionState,
    check_exit,
    scale_rails,
    vol_scale_from_returns,
)
from qqq_btc.qqq import config as qcfg

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from raw1s_rule_validation import (  # noqa: E402
    build_minute_frame,
    compute_oracle_edge,
    discover_raw1s_days,
    load_raw1s_bucket_day,
)

ENTRY_START = 15
ENTRY_END = 300


def _round4(x: float) -> float:
    return round(float(x), 4)


# ---------------------------------------------------------------------------
# 标签:逐 bar 反事实 rails 价值
# ---------------------------------------------------------------------------
def compute_rails_value(
    minute: pd.DataFrame,
    fill_model,
    rails_cfg: ExitRailsConfig,
    entry_delay_bars: int = 1,
) -> pd.DataFrame:
    """给 minute 增加 rails_value / rails_exit_reason / rails_hold / veto_dd15。"""
    out = minute.copy()
    n = len(out)
    bid = out["exec_call_bid"].to_numpy(dtype=np.float64)
    ask = out["exec_call_ask"].to_numpy(dtype=np.float64)
    mid = out["exec_call_mid"].to_numpy(dtype=np.float64)
    sbar = out["session_bar"].to_numpy(dtype=np.int64)
    mret = pd.Series(mid).pct_change().to_numpy()

    val = np.full(n, np.nan)
    hold = np.full(n, -1, dtype=np.int64)
    reasons: List[Optional[str]] = [None] * n
    veto = np.full(n, np.nan)  # 入场后 15bar 内最深 ROI(用于止损头/veto 头标签)

    for i in range(n):
        e = i + entry_delay_bars
        if e >= n - 1:
            continue
        ef = fill_model.entry_fill(bid[e], ask[e])
        if not (np.isfinite(ef) and ef > 0):
            continue
        drag = fill_model.commission_return_drag(ef)

        scale = vol_scale_from_returns(rails_cfg, mret[max(0, e - 90):e].tolist())
        cfg = scale_rails(rails_cfg, scale)

        pos = PositionState(entry_price=ef, entry_bar=e)
        exit_j = None
        reason = None
        min_roi15 = 0.0
        for j in range(e + 1, n):
            m = mid[j]
            if not (np.isfinite(m) and m > 0):
                continue
            if j - e <= 15:
                min_roi15 = min(min_roi15, m / ef - 1.0)
            r = check_exit(cfg, pos, m, j, session_bar_index=int(sbar[j]))
            if r is not None:
                exit_j = j
                reason = r
                break
        if exit_j is None:
            exit_j = n - 1
            reason = "DATA_END"
        xf = fill_model.exit_fill(bid[exit_j], ask[exit_j])
        if not (np.isfinite(xf) and xf > 0):
            continue
        val[i] = xf / ef - 1.0 - drag
        hold[i] = exit_j - e
        reasons[i] = reason
        veto[i] = min_roi15

    out["rails_value"] = val
    out["rails_hold"] = hold
    out["rails_exit_reason"] = reasons
    out["veto_dd15"] = veto
    return out


# ---------------------------------------------------------------------------
# 因果特征(仅用 bar i 及更早的信息)
# ---------------------------------------------------------------------------
FEATURES: List[str] = []

_TREND_X = (np.arange(30, dtype=np.float64) - 14.5) / np.arange(30, dtype=np.float64).std()


def _slope_of(arr: np.ndarray) -> float:
    if np.isnan(arr).any():
        return np.nan
    return float(np.dot(_TREND_X, arr - arr.mean()) / len(arr))


def _r2_of(arr: np.ndarray) -> float:
    if np.isnan(arr).any():
        return np.nan
    y = arr - arr.mean()
    denom = float(np.dot(y, y))
    if denom <= 0:
        return 0.0
    beta = float(np.dot(_TREND_X, y) / np.dot(_TREND_X, _TREND_X))
    resid = y - beta * _TREND_X
    return 1.0 - float(np.dot(resid, resid)) / denom


def attach_put_leg(minute: pd.DataFrame, put_ticks: pd.DataFrame) -> pd.DataFrame:
    """把 PUT 腿(bucket 0)的分钟 mid/spread/size 合并到主(CALL)分钟框。"""
    m = minute.copy()
    if put_ticks.empty:
        for c in ("put_mid", "put_spread_pct", "put_bid_size", "put_ask_size"):
            m[c] = np.nan
        return m
    pm = build_minute_frame(put_ticks)
    pm = pm[["timestamp", "exec_call_mid", "exec_call_spread_pct", "bid_size", "ask_size"]].rename(
        columns={
            "exec_call_mid": "put_mid",
            "exec_call_spread_pct": "put_spread_pct",
            "bid_size": "put_bid_size",
            "ask_size": "put_ask_size",
        }
    )
    m = m.merge(pm, on="timestamp", how="left")
    for c in ("put_mid", "put_spread_pct"):
        m[c] = m[c].ffill()
    return m


def build_causal_features(minute: pd.DataFrame) -> pd.DataFrame:
    m = minute.copy()
    mid = m["exec_call_mid"]
    logm = np.log(mid.where(mid > 0))

    feats = {}
    for w in (1, 3, 5, 10, 15, 30, 60):
        feats[f"ret_{w}m"] = logm.diff(w)
    for w in (10, 30, 60):
        feats[f"vol_{w}m"] = logm.diff().rolling(w, min_periods=max(3, w // 3)).std()
    feats["vol_ratio_10_60"] = feats["vol_10m"] / feats["vol_60m"]

    day_open = mid.iloc[0]
    feats["ret_from_open"] = logm - np.log(day_open) if day_open > 0 else logm * np.nan
    roll_max30 = mid.rolling(30, min_periods=5).max()
    roll_min30 = mid.rolling(30, min_periods=5).min()
    feats["dd_from_max30"] = mid / roll_max30 - 1.0
    feats["up_from_min30"] = mid / roll_min30 - 1.0
    feats["range_pos_30"] = (mid - roll_min30) / (roll_max30 - roll_min30)

    sp = m["exec_call_spread_pct"]
    feats["spread_pct"] = sp
    feats["spread_ma10"] = sp.rolling(10, min_periods=3).mean()
    feats["spread_z30"] = (sp - sp.rolling(30, min_periods=5).mean()) / (
        sp.rolling(30, min_periods=5).std() + 1e-9
    )

    bidsz = m.get("bid_size", pd.Series(0.0, index=m.index)).astype(float)
    asksz = m.get("ask_size", pd.Series(0.0, index=m.index)).astype(float)
    feats["size_imb"] = (bidsz - asksz) / (bidsz + asksz + 1e-9)
    feats["size_imb_ma10"] = pd.Series(feats["size_imb"]).rolling(10, min_periods=3).mean()

    sb = m["session_bar"].astype(float)
    feats["session_bar"] = sb
    feats["tod_sin"] = np.sin(2 * np.pi * sb / 390.0)
    feats["tod_cos"] = np.cos(2 * np.pi * sb / 390.0)

    # --- PUT 腿 + 合成现货(put-call parity:spot ≈ K + C - P,同 strike ATM) ---
    if "put_mid" in m.columns:
        strike = pd.to_numeric(m.get("strike"), errors="coerce")
        spot = strike + mid - m["put_mid"]
        logs = np.log(spot.where(spot > 0))
        for w in (1, 5, 15, 30, 60):
            feats[f"spot_ret_{w}m"] = logs.diff(w)
        feats["spot_vol_30m"] = logs.diff().rolling(30, min_periods=10).std()
        smax30 = spot.rolling(30, min_periods=5).max()
        smin30 = spot.rolling(30, min_periods=5).min()
        feats["spot_range_pos_30"] = (spot - smin30) / (smax30 - smin30)

        straddle = mid + m["put_mid"]  # IV/预期波动代理
        feats["straddle_ret_15m"] = np.log(straddle.where(straddle > 0)).diff(15)
        feats["cp_ratio"] = np.log((mid / m["put_mid"]).where((mid > 0) & (m["put_mid"] > 0)))
        feats["cp_ratio_chg_15m"] = pd.Series(feats["cp_ratio"]).diff(15)
        feats["put_spread_pct"] = m["put_spread_pct"]

        feats["spot_trend_slope_30"] = logs.rolling(30).apply(_slope_of, raw=True)
        feats["spot_trend_r2_30"] = logs.rolling(30).apply(_r2_of, raw=True)

    # 趋势拟合:过去 30bar 线性斜率 / R2(因果)
    feats["trend_slope_30"] = logm.rolling(30).apply(_slope_of, raw=True)
    feats["trend_r2_30"] = logm.rolling(30).apply(_r2_of, raw=True)

    for k, v in feats.items():
        m[k] = v
    global FEATURES
    FEATURES = sorted(feats.keys())
    return m


# ---------------------------------------------------------------------------
# 数据装载
# ---------------------------------------------------------------------------
def load_month_days(
    raw_dir: Path, symbol: str, globs: Sequence[str], bucket: int, put_bucket: int = 0
) -> List[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    out = []
    for g in globs:
        for fp in discover_raw1s_days(raw_dir, symbol, glob_pattern=g):
            date_str = fp.stem.split("_", 1)[-1]
            ticks = load_raw1s_bucket_day(fp, bucket)
            if ticks.empty:
                continue
            minute = build_minute_frame(ticks)
            if minute.empty or len(minute) < 120:
                continue
            minute = compute_rails_value(minute, qcfg.FILL_MODEL, qcfg.EXIT_RAILS)
            minute = compute_oracle_edge(minute, qcfg.FILL_MODEL, hold_bars=5)
            put_ticks = load_raw1s_bucket_day(fp, put_bucket)
            minute = attach_put_leg(minute, put_ticks)
            minute = build_causal_features(minute)
            tick_df = ticks[
                ["timestamp", "exec_call_bid", "exec_call_ask", "exec_call_spread_pct"]
            ]
            out.append((date_str, minute, tick_df))
    return out


def entry_mask(minute: pd.DataFrame) -> pd.Series:
    sb = minute["session_bar"].astype(int)
    return (sb >= ENTRY_START) & (sb <= ENTRY_END)


def to_xy(
    days, label_rank: bool = False, label_top: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    X, y, day_idx(入场窗、标签有效的行)。
    label_rank=True → y 为日内百分位;label_top=q → y 为「属于当日 top q」二值。
    """
    Xs, ys, ds = [], [], []
    for di, (_date, minute, _t) in enumerate(days):
        w = entry_mask(minute) & minute["rails_value"].notna()
        sub = minute.loc[w]
        y = sub["rails_value"]
        if label_top is not None:
            y = (y.rank(pct=True) >= 1.0 - label_top).astype(float)
        elif label_rank:
            y = y.rank(pct=True)
        Xs.append(sub[FEATURES].to_numpy(dtype=np.float64))
        ys.append(y.to_numpy(dtype=np.float64))
        ds.append(np.full(len(sub), di))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ds)


def daily_rank_ic(pred: np.ndarray, y: np.ndarray, day_idx: np.ndarray) -> List[float]:
    ics = []
    for d in np.unique(day_idx):
        sel = day_idx == d
        if sel.sum() < 30:
            continue
        rho, _ = spearmanr(pred[sel], y[sel])
        if np.isfinite(rho):
            ics.append(float(rho))
    return ics


# ---------------------------------------------------------------------------
# 因果 top-k replay:日内截至当前的分位门控
# ---------------------------------------------------------------------------
def causal_topk_signal(
    pred: np.ndarray,
    w: np.ndarray,
    top_pct: float,
    min_obs: int = 45,
    gate: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    分数 → 入场资格信号(因果):bar i 的预测若 >= 当日已见预测的 (1-top_pct)
    分位(且已积累 min_obs 个观测)则给 0.10,否则 -1。
    gate:布尔数组,False 的 bar 直接取消资格(仍计入分位历史)。
    """
    sig = np.full(len(pred), -1.0)
    hist: List[float] = []
    for i in range(len(pred)):
        if not w[i] or not np.isfinite(pred[i]):
            continue
        if len(hist) >= min_obs:
            thr = float(np.quantile(hist, 1.0 - top_pct))
            if pred[i] >= thr and (gate is None or bool(gate[i])):
                sig[i] = 0.10
        hist.append(float(pred[i]))
    return sig


def replay_month(
    days,
    preds_by_day: Dict[str, np.ndarray],
    top_pct: float,
    momentum_gate: Optional[float] = None,
) -> dict:
    day_rois, hits = [], []
    total_trades = 0
    exit_counts: Dict[str, int] = {}
    for date_str, minute, tick_df in days:
        pred = preds_by_day[date_str]
        w = entry_mask(minute).to_numpy()
        gate = None
        if momentum_gate is not None:
            # veto 接飞刀:入场 bar 的近 15m 权利金动量必须 > 阈值
            gate = (minute["ret_15m"].to_numpy() > momentum_gate)
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


def main() -> int:
    ap = argparse.ArgumentParser(description="rails-value label + LightGBM causal validation")
    ap.add_argument("--raw-1s-dir", default="/mnt/s990/data/raw_1s/dte1_options")
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--bucket", type=int, default=2)
    ap.add_argument("--train-globs", default="QQQ_2025-01-*.parquet,QQQ_2025-02-*.parquet,QQQ_2025-03-*.parquet,QQQ_2025-04-*.parquet")
    ap.add_argument("--val-globs", default="QQQ_2025-05-*.parquet")
    ap.add_argument("--test-globs", default="QQQ_2025-06-*.parquet")
    ap.add_argument("--top-pcts", default="0.02,0.05,0.10")
    ap.add_argument(
        "--label-rank",
        action="store_true",
        help="训练目标改为日内百分位排名(与噪声实验结论对齐:任务=日内排序)",
    )
    ap.add_argument(
        "--label-top",
        type=float,
        default=None,
        help="二分类目标:rails_value 属于当日 top X(如 0.10)则为 1",
    )
    ap.add_argument(
        "--out",
        default="New_Pro/baseline_qqq/reports/qqq_1dte_rails_value_lgbm.json",
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
    print("loading test ...")
    test_days = load_month_days(raw_dir, args.symbol, args.test_globs.split(","), args.bucket)
    print(f"  test days={len(test_days)}")

    # --- 标签质量诊断(train) ---
    all_v = np.concatenate([
        m.loc[entry_mask(m), "rails_value"].dropna().to_numpy() for _d, m, _t in train_days
    ])
    label_diag = {
        "n": int(all_v.size),
        "mean": _round4(float(all_v.mean())),
        "median": _round4(float(np.median(all_v))),
        "pos_frac": _round4(float((all_v > 0).mean())),
        "p10": _round4(float(np.quantile(all_v, 0.10))),
        "p90": _round4(float(np.quantile(all_v, 0.90))),
        "p99": _round4(float(np.quantile(all_v, 0.99))),
    }
    print(f"label diag: {label_diag}")

    Xtr, ytr, dtr = to_xy(train_days, label_rank=args.label_rank, label_top=args.label_top)
    Xva, yva, dva = to_xy(val_days, label_rank=args.label_rank, label_top=args.label_top)
    Xte, yte, dte = to_xy(test_days)  # test 始终对 raw rails_value 测 IC
    print(f"rows train={len(ytr)} val={len(yva)} test={len(yte)} feats={len(FEATURES)}")

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        Xtr,
        ytr,
        eval_set=[(Xva, yva)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    best_iter = model.best_iteration_ or model.n_estimators

    ic_tr = daily_rank_ic(model.predict(Xtr), ytr, dtr)
    ic_va = daily_rank_ic(model.predict(Xva), yva, dva)
    pred_te = model.predict(Xte)
    ic_te = daily_rank_ic(pred_te, yte, dte)
    print(
        f"rank IC  train={np.mean(ic_tr):+.3f}  val={np.mean(ic_va):+.3f}  "
        f"test={np.mean(ic_te):+.3f} (test day IC>0: {np.mean(np.array(ic_te) > 0):.0%})"
    )

    imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])

    # --- 因果 top-k replay(test 月;预测覆盖全部 bar,不只标签有效行) ---
    preds_by_day: Dict[str, np.ndarray] = {}
    for date_str, minute, _t in test_days:
        F = minute[FEATURES].to_numpy(dtype=np.float64)
        preds_by_day[date_str] = model.predict(F)

    replay_rows = []
    for pct in top_pcts:
        for mg in (None, -0.02, 0.0):
            rr = replay_month(test_days, preds_by_day, pct, momentum_gate=mg)
            rr["momentum_gate"] = mg
            replay_rows.append(rr)
            tag = "no_gate" if mg is None else f"mom>{mg:+.2f}"
            print(
                f"replay top{pct:.0%} [{tag}]: win={rr['win_days']}/{rr['days']} "
                f"dayROI={rr['day_roi_mean']:+.1%} comp={rr['compound']:+.1%} "
                f"trades={rr['trades']} hit={rr['hit_rate']:.0%} worst_day={rr['worst_day']:+.1%}"
            )

    result = {
        "meta": {
            "raw_1s_dir": str(raw_dir),
            "bucket": args.bucket,
            "train_globs": args.train_globs,
            "val_globs": args.val_globs,
            "test_globs": args.test_globs,
            "features": FEATURES,
            "best_iteration": int(best_iter),
            "anchor_2025m06": {
                "oracle_continuous_compound": 38.06,
                "oracle_rank_top5_compound": 1421.79,
                "pure_noise_compound": -0.505,
            },
        },
        "label_diag": label_diag,
        "rank_ic": {
            "train_mean": _round4(float(np.mean(ic_tr))),
            "val_mean": _round4(float(np.mean(ic_va))),
            "test_mean": _round4(float(np.mean(ic_te))),
            "test_daily": [_round4(v) for v in ic_te],
            "test_pos_day_frac": _round4(float(np.mean(np.array(ic_te) > 0))),
        },
        "feature_importance_top15": [[k, int(v)] for k, v in imp[:15]],
        "test_replay": replay_rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
