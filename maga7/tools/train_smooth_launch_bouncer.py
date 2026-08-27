#!/usr/bin/env python3
"""Train shared LGBM bouncer on smooth/impulse launch candidates.

Pipeline:
  1) Fire dual-sleeve launches (high recall)
  2) Label true/false from forward stock MFE/MAE
  3) Walk-forward LGBM: P(allow)
  4) Scoreboard: precision@threshold + gated stock UP trail120 replay

Model role = veto false starts — not Call/Put prediction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.lgbm_bouncer import extract_bouncer_features, save_lgbm_model
from maga7.common.signals import attach_mf_features, load_stock_month_files
from maga7.common.smooth_trend import (
    ImpulseLaunchConfig,
    SmoothLaunchConfig,
    SmoothStockTradeConfig,
    apply_day_portfolio_cap,
    detect_impulse_launches_day,
    detect_smooth_launches_day,
    merge_dual_sleeve_launches,
    replay_smooth_impulse_stock_day,
)
from maga7.tools.run_smooth_impulse_stock_replay import SYMS, _equity

NY = "America/New_York"

# Bake-off winner (D): keep only features with OOS gain; drop streak/mf dead weight.
MONTHS = [f"2026-{m:02d}" for m in range(1, 8)]
LAUNCH_FEATS = (
    "look_ret",
    "path_eff",
    "up_frac",
    "max_dd",
    "from_extreme",
    "score",
)
USEFUL_FEATS = (
    "look_ret",
    "path_eff",
    "from_extreme",
    "score",
    "bounce_lod",
    "tod_min",
    "vol_z",
    "qqq_gap_open",
    "qqq_from_prev",
    "gap_open",
    "from_prev",
    "max_dd",
    "up_frac",
)


def _fwd_mfe_mae(
    day: pd.DataFrame,
    *,
    entry_ts: pd.Timestamp,
    direction: str,
    horizon_minutes: int = 90,
) -> tuple[float | None, float | None]:
    d = day.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    if d["timestamp"].dt.tz is None:
        d["timestamp"] = d["timestamp"].dt.tz_localize(NY)
    else:
        d["timestamp"] = d["timestamp"].dt.tz_convert(NY)
    et = pd.Timestamp(entry_ts)
    if et.tzinfo is None:
        et = et.tz_localize(NY)
    else:
        et = et.tz_convert(NY)
    after = d[d.timestamp >= et]
    if after.empty:
        return None, None
    px0 = float(after.iloc[0]["close"])
    if px0 <= 0:
        return None, None
    fut = after[after.timestamp <= et + pd.Timedelta(minutes=horizon_minutes)]
    if len(fut) < 5:
        return None, None
    # use high/low if present else close
    if "high" in fut.columns and "low" in fut.columns:
        hi = pd.to_numeric(fut["high"], errors="coerce")
        lo = pd.to_numeric(fut["low"], errors="coerce")
        if direction == "UP":
            mfe = float(hi.max() / px0 - 1.0)
            mae = float(lo.min() / px0 - 1.0)  # negative
            mae = -mae  # adverse magnitude
        else:
            mfe = float(1.0 - lo.min() / px0)
            mae = float(hi.max() / px0 - 1.0)
    else:
        c = pd.to_numeric(fut["close"], errors="coerce")
        if direction == "UP":
            mfe = float(c.max() / px0 - 1.0)
            mae = float(-(c.min() / px0 - 1.0))
        else:
            mfe = float(1.0 - c.min() / px0)
            mae = float(c.max() / px0 - 1.0)
    return mfe, max(0.0, mae)


def build_dataset(
    data: dict[str, pd.DataFrame],
    *,
    smooth_cfg: SmoothLaunchConfig,
    impulse_cfg: ImpulseLaunchConfig,
    good_mfe: float,
    toxic_mae: float,
    horizon: int,
) -> pd.DataFrame:
    qqq = data.get("QQQ")
    rows: list[dict] = []
    for sym in SYMS:
        raw = data.get(sym)
        if raw is None or raw.empty:
            continue
        print(f"[dataset] {sym}", flush=True)
        for date in sorted(raw["date"].astype(str).unique()):
            day = raw[raw["date"].astype(str) == date]
            qday = qqq[qqq["date"].astype(str) == date] if qqq is not None else None
            smooth = detect_smooth_launches_day(
                day, symbol=sym, date=date, cfg=smooth_cfg, directions=("UP", "DN")
            )
            impulse = detect_impulse_launches_day(
                day, symbol=sym, date=date, cfg=impulse_cfg, directions=("UP", "DN")
            )
            merged = merge_dual_sleeve_launches(
                smooth, impulse, first_per_symbol_dir=False, prefer_smooth=True
            )
            for ln, sleeve in merged:
                mfe, mae = _fwd_mfe_mae(
                    day, entry_ts=ln.detect_ts, direction=ln.direction, horizon_minutes=horizon
                )
                if mfe is None or mae is None:
                    continue
                # true start: enough favorable excursion without deep adverse
                y_allow = int(mfe >= good_mfe and mae <= toxic_mae)
                y_toxic = int(mae >= toxic_mae and mfe < good_mfe)
                feat = extract_bouncer_features(
                    symbol=sym,
                    direction=ln.direction,
                    asof_ts=ln.detect_ts,
                    stock_df=day,
                    qqq_df=qday,
                )
                if feat is None:
                    continue
                row = {
                    "date": date,
                    "symbol": sym,
                    "direction": ln.direction,
                    "detect_ts": str(ln.detect_ts),
                    "sleeve": sleeve,
                    "look_ret": ln.look_ret,
                    "path_eff": ln.path_eff,
                    "up_frac": ln.up_frac,
                    "max_dd": ln.max_dd,
                    "from_extreme": ln.from_extreme,
                    "score": ln.score,
                    "sleeve_smooth": 1.0 if sleeve == "smooth" else 0.0,
                    "hour": float(ln.detect_ts.hour),
                    "minute": float(ln.detect_ts.minute),
                    "mfe": mfe,
                    "mae": mae,
                    "y_allow": y_allow,
                    "y_toxic": y_toxic,
                    "label_src": "smooth_launch_stock",
                    **feat,
                }
                rows.append(row)
    return pd.DataFrame(rows)


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y = y_true.astype(float)
    if len(np.unique(y)) < 2:
        return None
    order = np.argsort(y_score)
    y = y[order]
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos <= 0 or n_neg <= 0:
        return None
    ranks = np.arange(1, len(y) + 1, dtype=float)
    sum_ranks_pos = ranks[y > 0.5].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default=(
            "maga7/CONFIG/strategy_profiles/"
            "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
        ),
    )
    ap.add_argument("--start-date", default="2026-01-01")
    ap.add_argument("--end-date", default="2026-07-17")
    ap.add_argument("--train-end", default="2026-04-30")
    ap.add_argument("--valid-start", default="2026-05-01")
    ap.add_argument("--good-mfe", type=float, default=0.015)
    ap.add_argument("--toxic-mae", type=float, default=0.008)
    ap.add_argument("--horizon", type=int, default=90)
    ap.add_argument("--p-min", type=float, default=0.55)
    ap.add_argument("--up-only", action="store_true", default=True)
    ap.add_argument("--both-dirs", action="store_true", help="Train on UP+DN (overrides --up-only)")
    ap.add_argument(
        "--all-feats",
        action="store_true",
        help="Use full feature set (default: bake-off useful feats only)",
    )
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/smooth_launch_bouncer_promoted_v1",
    )
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    import lightgbm as lgb

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prof = load_profile(args.profile)
    root = Path(prof["_paths"]["stock_root"]).expanduser()

    print("[load] stocks", flush=True)
    data: dict[str, pd.DataFrame] = {}
    for sym in SYMS + ["QQQ"]:
        raw = load_stock_month_files(root, sym, MONTHS)
        if raw.empty:
            continue
        raw = attach_mf_features(raw)
        data[sym] = raw[raw["date"].astype(str).between(args.start_date, args.end_date)]

    smooth_cfg = SmoothLaunchConfig(scan_end="11:30", min_look_ret=0.002, cooldown_minutes=60)
    impulse_cfg = ImpulseLaunchConfig(scan_end="11:30", min_look_ret=0.004)

    ds = build_dataset(
        data,
        smooth_cfg=smooth_cfg,
        impulse_cfg=impulse_cfg,
        good_mfe=float(args.good_mfe),
        toxic_mae=float(args.toxic_mae),
        horizon=int(args.horizon),
    )
    if ds.empty:
        raise SystemExit("empty dataset")
    ds_path = out / "dataset_all.parquet"
    ds.to_parquet(ds_path, index=False)
    print(
        f"dataset n={len(ds)} allow_rate={ds.y_allow.mean():.3f} "
        f"UP={int((ds.direction=='UP').sum())} DN={int((ds.direction=='DN').sum())}",
        flush=True,
    )

    up_only = bool(args.up_only) and not bool(args.both_dirs)
    if up_only:
        ds = ds[ds["direction"].astype(str).str.upper() == "UP"].reset_index(drop=True)
        print(f"train subset UP-only n={len(ds)} allow_rate={ds.y_allow.mean():.3f}", flush=True)

    # features: default = bake-off useful set; --all-feats for full
    from maga7.common.lgbm_bouncer import FEATURE_COLS

    if args.all_feats:
        feat_cols = [c for c in list(FEATURE_COLS) + list(LAUNCH_FEATS) if c in ds.columns]
    else:
        feat_cols = [c for c in USEFUL_FEATS if c in ds.columns]
    if up_only and "dir_sign" in feat_cols:
        feat_cols = [c for c in feat_cols if c != "dir_sign"]
    dates = ds["date"].astype(str)
    tr = dates <= str(args.train_end)
    va = dates >= str(args.valid_start)
    if int(tr.sum()) < 50 or int(va.sum()) < 20:
        raise SystemExit(f"split too small train={tr.sum()} valid={va.sum()}")

    # train P(allow) directly
    y_tr = ds.loc[tr, "y_allow"].astype(int).to_numpy()
    y_va = ds.loc[va, "y_allow"].astype(int).to_numpy()
    X_tr = ds.loc[tr, feat_cols].astype(float).to_numpy()
    X_va = ds.loc[va, feat_cols].astype(float).to_numpy()
    # No scale_pos_weight: keep scores rank-usable; gate by quantile, not 0.55.
    train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=list(feat_cols))
    valid_set = lgb.Dataset(X_va, label=y_va, feature_name=list(feat_cols), reference=train_set)
    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 15,
        "lambda_l1": 0.05,
        "lambda_l2": 0.5,
        "verbosity": -1,
        "seed": int(args.seed),
    }
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    p_tr = booster.predict(X_tr)
    p_va = booster.predict(X_va)
    model_path = out / "lgbm_smooth_launch_bouncer.txt"
    ds.to_parquet(out / "dataset.parquet", index=False)
    meta = {
        "feature_cols": feat_cols,
        "target": "allow",
        "up_only": up_only,
        "train_end": args.train_end,
        "valid_start": args.valid_start,
        "good_mfe": args.good_mfe,
        "toxic_mae": args.toxic_mae,
        "horizon": args.horizon,
        "auc_train": _auc(y_tr, p_tr),
        "auc_valid": _auc(y_va, p_va),
        "allow_rate_train": float(y_tr.mean()),
        "allow_rate_valid": float(y_va.mean()),
        "n_train": int(tr.sum()),
        "n_valid": int(va.sum()),
        "p_train_range": [float(np.min(p_tr)), float(np.max(p_tr))],
        "p_valid_range": [float(np.min(p_va)), float(np.max(p_va))],
        "best_iteration": int(getattr(booster, "best_iteration", 0) or 0),
    }
    save_lgbm_model(booster, model_path, meta=meta)
    print("AUC train/valid", meta["auc_train"], meta["auc_valid"], flush=True)

    # threshold sweep on valid — absolute + quantile (scores often compressed)
    va_df = ds.loc[va].copy()
    va_df["p_allow"] = p_va
    base = float(va_df.y_allow.mean())
    sweeps = []
    for thr in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        sel = va_df["p_allow"] >= thr
        n = int(sel.sum())
        if n == 0:
            sweeps.append({"mode": "abs", "p_min": thr, "n": 0, "precision": None, "recall": None, "allow_base": base})
            continue
        prec = float(va_df.loc[sel, "y_allow"].mean())
        rec = float(va_df.loc[sel, "y_allow"].sum() / max(va_df["y_allow"].sum(), 1))
        sweeps.append({"mode": "abs", "p_min": thr, "n": n, "precision": prec, "recall": rec, "allow_base": base})
    # quantile gates on valid ranks (train-calibrated thr also recorded)
    for q in [0.50, 0.60, 0.70, 0.80, 0.90]:
        thr = float(np.quantile(p_va, q))
        sel = va_df["p_allow"] >= thr
        n = int(sel.sum())
        prec = float(va_df.loc[sel, "y_allow"].mean()) if n else None
        rec = float(va_df.loc[sel, "y_allow"].sum() / max(va_df["y_allow"].sum(), 1)) if n else None
        sweeps.append(
            {
                "mode": f"q{q:.2f}",
                "p_min": thr,
                "n": n,
                "precision": prec,
                "recall": rec,
                "allow_base": base,
            }
        )
    # UP-only quantile
    va_up = va_df[va_df.direction == "UP"]
    for q in [0.50, 0.70, 0.80, 0.90]:
        if va_up.empty:
            break
        thr = float(np.quantile(va_up["p_allow"].to_numpy(), q))
        sel = va_up["p_allow"] >= thr
        n = int(sel.sum())
        prec = float(va_up.loc[sel, "y_allow"].mean()) if n else None
        rec = float(va_up.loc[sel, "y_allow"].sum() / max(va_up["y_allow"].sum(), 1)) if n else None
        sweeps.append(
            {
                "mode": f"UP_q{q:.2f}",
                "p_min": thr,
                "n": n,
                "precision": prec,
                "recall": rec,
                "allow_base": float(va_up.y_allow.mean()),
            }
        )
    sw = pd.DataFrame(sweeps)
    sw.to_csv(out / "threshold_sweep.csv", index=False)
    print(sw.to_string(index=False), flush=True)

    # pick thr: among quantile modes with n>=30, max lift*f1
    usable = sw[(sw["n"] >= 30) & sw["precision"].notna() & sw["mode"].astype(str).str.startswith("q")].copy()
    if len(usable):
        usable["lift"] = usable["precision"] / usable["allow_base"].clip(lower=1e-6)
        usable["f1"] = usable.apply(
            lambda r: 2 * r.precision * r.recall / (r.precision + r.recall)
            if r.precision and r.recall and (r.precision + r.recall) > 0
            else 0.0,
            axis=1,
        )
        # prefer precision lift >= 1.25 else best f1
        hi = usable[usable["lift"] >= 1.25]
        best_thr = float((hi if len(hi) else usable).sort_values(["lift", "f1"], ascending=False).iloc[0]["p_min"])
    else:
        best_thr = float(np.quantile(p_va, 0.70))

    # Stock backtest on valid window: ungated vs gated (score live)
    trade_cfg = SmoothStockTradeConfig(
        break_max_adverse=0.012,
        max_hold_minutes=180,
        break_min_up_frac=0.35,
        first_per_symbol_dir=True,
    )
    qqq = data.get("QQQ")

    def _score_launch(day: pd.DataFrame, qday: pd.DataFrame | None, ln, sleeve: str) -> float | None:
        feat = extract_bouncer_features(
            symbol=ln.symbol,
            direction=ln.direction,
            asof_ts=ln.detect_ts,
            stock_df=day,
            qqq_df=qday,
        )
        if feat is None:
            return None
        row = {
            **feat,
            "look_ret": ln.look_ret,
            "path_eff": ln.path_eff,
            "up_frac": ln.up_frac,
            "max_dd": ln.max_dd,
            "from_extreme": ln.from_extreme,
            "score": ln.score,
            "sleeve_smooth": 1.0 if sleeve == "smooth" else 0.0,
            "hour": float(ln.detect_ts.hour),
            "minute": float(ln.detect_ts.minute),
        }
        x = np.array([[float(row.get(c, 0.0)) for c in feat_cols]], dtype=float)
        return float(booster.predict(x)[0])

    def run_stock(gated: bool, p_min: float, tag: str) -> dict:
        all_trades = []
        for sym in SYMS:
            raw = data.get(sym)
            if raw is None:
                continue
            dates = sorted(
                d for d in raw["date"].astype(str).unique() if d >= str(args.valid_start)
            )
            for date in dates:
                day = raw[raw["date"].astype(str) == date]
                qday = qqq[qqq["date"].astype(str) == date] if qqq is not None else None
                rows = replay_smooth_impulse_stock_day(
                    day,
                    symbol=sym,
                    date=date,
                    smooth_cfg=smooth_cfg,
                    impulse_cfg=impulse_cfg,
                    trade_cfg=trade_cfg,
                )
                rows = [r for r in rows if r["direction"] == "UP"]
                if gated:
                    # re-detect to get SmoothLaunch objects for scoring
                    smooth = detect_smooth_launches_day(
                        day, symbol=sym, date=date, cfg=smooth_cfg, directions=("UP",)
                    )
                    impulse = detect_impulse_launches_day(
                        day, symbol=sym, date=date, cfg=impulse_cfg, directions=("UP",)
                    )
                    merged = merge_dual_sleeve_launches(
                        smooth, impulse, first_per_symbol_dir=True, prefer_smooth=True
                    )
                    p_by_ts = {}
                    for ln, sleeve in merged:
                        p = _score_launch(day, qday, ln, sleeve)
                        if p is not None:
                            p_by_ts[str(ln.detect_ts)] = p
                    kept = []
                    for r in rows:
                        p = p_by_ts.get(str(r["detect_ts"]))
                        if p is None:
                            # closest ts within 2 minutes
                            rt = pd.Timestamp(r["detect_ts"])
                            if p_by_ts:
                                best = min(
                                    p_by_ts.items(),
                                    key=lambda kv: abs((pd.Timestamp(kv[0]) - rt).total_seconds()),
                                )
                                if abs((pd.Timestamp(best[0]) - rt).total_seconds()) <= 120:
                                    p = best[1]
                        if p is not None and p >= p_min:
                            r = dict(r)
                            r["p_allow"] = p
                            kept.append(r)
                    rows = kept
                all_trades.extend(rows)
        capped = apply_day_portfolio_cap(all_trades, max_positions=2)
        tdf = pd.DataFrame(capped)
        if not tdf.empty:
            tdf.to_csv(out / f"trades_{tag}.csv", index=False)
        eq = (
            _equity(tdf, frac=0.5)
            if not tdf.empty
            else {"total_ret": 0, "maxdd": 0, "n_trades": 0, "trade_win": None, "avg_trade_ret": None}
        )
        return {
            "gated": gated,
            "p_min": p_min if gated else None,
            "total_ret": eq.get("total_ret"),
            "maxdd": eq.get("maxdd"),
            "n_trades": eq.get("n_trades"),
            "trade_win": eq.get("trade_win"),
            "avg_trade_ret": eq.get("avg_trade_ret"),
        }

    # gate thresholds: best + common quantiles of valid scores
    thr_set = {best_thr}
    for q in (0.50, 0.70, 0.80, 0.90):
        thr_set.add(float(np.quantile(p_va, q)))
    bt_rows = [run_stock(False, 0.0, "ungated")]
    for thr in sorted(thr_set):
        bt_rows.append(run_stock(True, float(thr), f"gated_p{thr:.4f}"))
    btdf = pd.DataFrame(bt_rows)
    btdf.to_csv(out / "backtest_gate.csv", index=False)

    summary = {
        "meta": meta,
        "best_thr": best_thr,
        "threshold_sweep": sweeps,
        "backtest_valid_window": btdf.to_dict(orient="records"),
        "model_path": str(model_path),
        "dataset_path": str(ds_path),
        "note": "Shared LGBM on launch candidates; label=stock MFE/MAE path; veto only.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # importance
    imp = pd.DataFrame(
        {"feature": feat_cols, "gain": booster.feature_importance(importance_type="gain")}
    ).sort_values("gain", ascending=False)
    imp.to_csv(out / "feature_importance.csv", index=False)

    def _tbl(df: pd.DataFrame) -> str:
        return df.to_string(index=False)

    lines = [
        "# Smooth Launch LGBM Bouncer",
        "",
        f"**Valid AUC: `{meta['auc_valid']}`** · train AUC `{meta['auc_train']}`",
        f"Allow rate train/valid: `{meta['allow_rate_train']:.2%}` / `{meta['allow_rate_valid']:.2%}`",
        f"Chosen p_min ≈ `{best_thr}`",
        "",
        "## Threshold sweep (valid)",
        "",
        "```",
        _tbl(sw),
        "```",
        "",
        "## Stock UP trail120 on valid (ungated vs gated)",
        "",
        "```",
        _tbl(btdf),
        "```",
        "",
        "## Top features",
        "",
        "```",
        _tbl(imp.head(15)),
        "```",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(btdf.to_string(index=False), flush=True)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
