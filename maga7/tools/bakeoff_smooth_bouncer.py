#!/usr/bin/env python3
"""Bake-off: keep whatever actually lifts valid stock UP trail120.

Variants:
  A short_all_mfe10     — May–midJun train, all feats (prior baseline)
  B long_all_mfe10      — Jan–Apr train, May–Jul valid, all feats
  C long_useful_mfe10   — long train, top features only
  D long_useful_mfe15   — long train, useful feats, stricter MFE≥1.5%
  E rule_useful         — no ML; train-tuned rule on useful launch/session feats
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
from maga7.common.lgbm_bouncer import FEATURE_COLS, extract_bouncer_features, save_lgbm_model
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
from maga7.tools.train_smooth_launch_bouncer import LAUNCH_FEATS, _auc, _fwd_mfe_mae, build_dataset

NY = "America/New_York"

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

MONTHS_LONG = [f"2026-{m:02d}" for m in range(1, 8)]


def _train_lgbm(X_tr, y_tr, X_va, y_va, feat_cols, seed: int = 7):
    import lightgbm as lgb

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
        "min_data_in_leaf": 20,
        "lambda_l1": 0.05,
        "lambda_l2": 0.5,
        "verbosity": -1,
        "seed": seed,
    }
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    return booster


def _score_row(booster, feat_cols, row: dict) -> float:
    x = np.array([[float(row.get(c, 0.0)) for c in feat_cols]], dtype=float)
    return float(booster.predict(x)[0])


def _rule_score(row: pd.Series) -> float:
    """Simple additive score on useful feats (higher = more allow)."""
    return (
        100.0 * float(row.get("look_ret") or 0)
        + 2.0 * float(row.get("path_eff") or 0)
        + 50.0 * float(row.get("bounce_lod") or 0)
        + 30.0 * float(row.get("from_extreme") or 0)
        - 0.01 * abs(float(row.get("tod_min") or 600) - 600)  # prefer ~10:00
        + 20.0 * float(row.get("qqq_gap_open") or 0)
    )


def run_stock_gate(
    data: dict[str, pd.DataFrame],
    *,
    valid_start: str,
    smooth_cfg: SmoothLaunchConfig,
    impulse_cfg: ImpulseLaunchConfig,
    trade_cfg: SmoothStockTradeConfig,
    scorer,
    p_min: float | None,
) -> dict:
    qqq = data.get("QQQ")
    all_trades = []
    for sym in SYMS:
        raw = data.get(sym)
        if raw is None:
            continue
        dates = sorted(d for d in raw["date"].astype(str).unique() if d >= valid_start)
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
            if p_min is not None:
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
                    p = scorer(day, qday, ln, sleeve)
                    if p is not None:
                        p_by_ts[str(ln.detect_ts)] = p
                kept = []
                for r in rows:
                    p = p_by_ts.get(str(r["detect_ts"]))
                    if p is None and p_by_ts:
                        rt = pd.Timestamp(r["detect_ts"])
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
    eq = (
        _equity(tdf, frac=0.5)
        if not tdf.empty
        else {"total_ret": 0, "maxdd": 0, "n_trades": 0, "trade_win": None, "avg_trade_ret": None}
    )
    return {
        "total_ret": eq.get("total_ret"),
        "maxdd": eq.get("maxdd"),
        "n_trades": eq.get("n_trades"),
        "trade_win": eq.get("trade_win"),
        "avg_trade_ret": eq.get("avg_trade_ret"),
        "trades": tdf,
    }


def eval_variant(
    name: str,
    ds: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    *,
    train_end: str,
    calib_start: str,
    calib_end: str,
    bt_start: str,
    feat_cols: list[str],
    mode: str,
    out: Path,
    smooth_cfg,
    impulse_cfg,
    trade_cfg,
) -> dict:
    dates = ds["date"].astype(str)
    tr = dates <= train_end
    va = (dates >= calib_start) & (dates <= calib_end)
    if int(tr.sum()) < 40 or int(va.sum()) < 20:
        return {"name": name, "error": f"split too small tr={tr.sum()} va={va.sum()}"}

    y_tr = ds.loc[tr, "y_allow"].astype(int).to_numpy()
    y_va = ds.loc[va, "y_allow"].astype(int).to_numpy()
    base = float(y_va.mean()) if len(y_va) else None

    if mode == "lgbm":
        X_tr = ds.loc[tr, feat_cols].astype(float).to_numpy()
        X_va = ds.loc[va, feat_cols].astype(float).to_numpy()
        booster = _train_lgbm(X_tr, y_tr, X_va, y_va, feat_cols)
        p_tr = booster.predict(X_tr)
        p_va = booster.predict(X_va)
        auc_tr, auc_va = _auc(y_tr, p_tr), _auc(y_va, p_va)
        # pick q among 0.5/0.7/0.8/0.9 maximizing lift with n>=20
        best = None
        for q in (0.50, 0.70, 0.80, 0.90):
            thr = float(np.quantile(p_va, q))
            sel = p_va >= thr
            n = int(sel.sum())
            if n < 20:
                continue
            prec = float(y_va[sel].mean())
            rec = float(y_va[sel].sum() / max(y_va.sum(), 1))
            lift = prec / max(base or 1e-6, 1e-6)
            cand = {"q": q, "thr": thr, "n": n, "precision": prec, "recall": rec, "lift": lift}
            if best is None or (lift, prec) > (best["lift"], best["precision"]):
                best = cand
        if best is None:
            best = {"q": 0.7, "thr": float(np.quantile(p_va, 0.7)), "n": 0, "precision": None, "recall": None, "lift": 0}
        save_lgbm_model(booster, out / f"{name}_model.txt", meta={"feature_cols": feat_cols, "variant": name, **best})

        def scorer(day, qday, ln, sleeve):
            feat = extract_bouncer_features(
                symbol=ln.symbol, direction=ln.direction, asof_ts=ln.detect_ts, stock_df=day, qqq_df=qday
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
            return _score_row(booster, feat_cols, row)

        thr = float(best["thr"])
    else:
        # rule mode
        tr_df = ds.loc[tr].copy()
        va_df = ds.loc[va].copy()
        tr_df["rule"] = tr_df.apply(_rule_score, axis=1)
        va_df["rule"] = va_df.apply(_rule_score, axis=1)
        p_tr = tr_df["rule"].to_numpy()
        p_va = va_df["rule"].to_numpy()
        auc_tr, auc_va = _auc(y_tr, p_tr), _auc(y_va, p_va)
        best = None
        for q in (0.50, 0.70, 0.80, 0.90):
            thr = float(np.quantile(p_va, q))
            sel = p_va >= thr
            n = int(sel.sum())
            if n < 20:
                continue
            prec = float(y_va[sel].mean())
            rec = float(y_va[sel].sum() / max(y_va.sum(), 1))
            lift = prec / max(base or 1e-6, 1e-6)
            cand = {"q": q, "thr": thr, "n": n, "precision": prec, "recall": rec, "lift": lift}
            if best is None or (lift, prec) > (best["lift"], best["precision"]):
                best = cand
        if best is None:
            best = {"q": 0.7, "thr": float(np.quantile(p_va, 0.7)), "n": 0, "precision": None, "recall": None, "lift": 0}

        def scorer(day, qday, ln, sleeve):
            feat = extract_bouncer_features(
                symbol=ln.symbol, direction=ln.direction, asof_ts=ln.detect_ts, stock_df=day, qqq_df=qday
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
            }
            return _rule_score(pd.Series(row))

        thr = float(best["thr"])

    ung = run_stock_gate(
        data,
        valid_start=bt_start,
        smooth_cfg=smooth_cfg,
        impulse_cfg=impulse_cfg,
        trade_cfg=trade_cfg,
        scorer=scorer,
        p_min=None,
    )
    gat = run_stock_gate(
        data,
        valid_start=bt_start,
        smooth_cfg=smooth_cfg,
        impulse_cfg=impulse_cfg,
        trade_cfg=trade_cfg,
        scorer=scorer,
        p_min=thr,
    )
    if not gat["trades"].empty:
        gat["trades"].to_csv(out / f"trades_{name}_gated.csv", index=False)

    return {
        "name": name,
        "mode": mode,
        "n_train": int(tr.sum()),
        "n_valid": int(va.sum()),
        "allow_base": base,
        "auc_train": auc_tr,
        "auc_valid": auc_va,
        "gate_q": best.get("q"),
        "gate_thr": thr,
        "gate_precision": best.get("precision"),
        "gate_recall": best.get("recall"),
        "gate_lift": best.get("lift"),
        "ungated_ret": ung["total_ret"],
        "ungated_maxdd": ung["maxdd"],
        "ungated_n": ung["n_trades"],
        "ungated_win": ung["trade_win"],
        "gated_ret": gat["total_ret"],
        "gated_maxdd": gat["maxdd"],
        "gated_n": gat["n_trades"],
        "gated_win": gat["trade_win"],
        "gated_avg": gat["avg_trade_ret"],
        "delta_ret": (gat["total_ret"] or 0) - (ung["total_ret"] or 0),
        "delta_win": ((gat["trade_win"] or 0) - (ung["trade_win"] or 0)) if gat["trade_win"] is not None else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default=(
            "maga7/CONFIG/strategy_profiles/"
            "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
        ),
    )
    ap.add_argument("--out", default="/mnt/s990/data/maga7/results/smooth_bouncer_bakeoff_v1")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prof = load_profile(args.profile)
    root = Path(prof["_paths"]["stock_root"]).expanduser()

    print("[load] stocks Jan–Jul 2026", flush=True)
    data: dict[str, pd.DataFrame] = {}
    for sym in SYMS + ["QQQ"]:
        raw = load_stock_month_files(root, sym, MONTHS_LONG)
        if raw.empty:
            continue
        raw = attach_mf_features(raw)
        data[sym] = raw[raw["date"].astype(str).between("2026-01-01", "2026-07-17")]

    smooth_cfg = SmoothLaunchConfig(scan_end="11:30", min_look_ret=0.002, cooldown_minutes=60)
    impulse_cfg = ImpulseLaunchConfig(scan_end="11:30", min_look_ret=0.004)
    trade_cfg = SmoothStockTradeConfig(
        break_max_adverse=0.012,
        max_hold_minutes=180,
        break_min_up_frac=0.35,
        first_per_symbol_dir=True,
    )

    print("[dataset] mfe10", flush=True)
    ds10 = build_dataset(
        data,
        smooth_cfg=smooth_cfg,
        impulse_cfg=impulse_cfg,
        good_mfe=0.010,
        toxic_mae=0.008,
        horizon=90,
    )
    ds10 = ds10[ds10.direction == "UP"].reset_index(drop=True)
    ds10.to_parquet(out / "dataset_up_mfe10.parquet", index=False)
    print(f"  n={len(ds10)} allow={ds10.y_allow.mean():.3f}", flush=True)

    print("[dataset] mfe15", flush=True)
    # rebuild labels on same launches for stricter threshold
    ds15 = ds10.copy()
    ds15["y_allow"] = ((ds15["mfe"] >= 0.015) & (ds15["mae"] <= 0.008)).astype(int)
    ds15.to_parquet(out / "dataset_up_mfe15.parquet", index=False)
    print(f"  n={len(ds15)} allow={ds15.y_allow.mean():.3f}", flush=True)

    all_feats = [c for c in list(FEATURE_COLS) + list(LAUNCH_FEATS) if c in ds10.columns and c != "dir_sign"]
    useful = [c for c in USEFUL_FEATS if c in ds10.columns]

    # Common OOS stock window Jun16–Jul17. Calibrate thr on held-out May–Jun15 when possible.
    bt_start = "2026-06-16"
    variants = [
        # short: train May1–Jun15, calib=train (optimistic thr), BT Jun16+
        ("A_short_all_mfe10", ds10, "2026-06-15", "2026-05-01", "2026-06-15", all_feats, "lgbm"),
        # long: train Jan–Apr, calib May–Jun15, BT Jun16+
        ("B_long_all_mfe10", ds10, "2026-04-30", "2026-05-01", "2026-06-15", all_feats, "lgbm"),
        ("C_long_useful_mfe10", ds10, "2026-04-30", "2026-05-01", "2026-06-15", useful, "lgbm"),
        ("D_long_useful_mfe15", ds15, "2026-04-30", "2026-05-01", "2026-06-15", useful, "lgbm"),
        ("E_rule_useful_mfe10", ds10, "2026-04-30", "2026-05-01", "2026-06-15", useful, "rule"),
    ]

    rows = []
    for name, ds, train_end, calib_start, calib_end, feats, mode in variants:
        print(f"[eval] {name}", flush=True)
        r = eval_variant(
            name,
            ds,
            data,
            train_end=train_end,
            calib_start=calib_start,
            calib_end=calib_end,
            bt_start=bt_start,
            feat_cols=feats,
            mode=mode,
            out=out,
            smooth_cfg=smooth_cfg,
            impulse_cfg=impulse_cfg,
            trade_cfg=trade_cfg,
        )
        rows.append(r)
        print(
            f"  auc_va={r.get('auc_valid')} lift={r.get('gate_lift')} "
            f"gated_ret={r.get('gated_ret')} win={r.get('gated_win')} n={r.get('gated_n')}",
            flush=True,
        )

    rdf = pd.DataFrame(rows)
    rdf.to_csv(out / "bakeoff.csv", index=False)

    # pick winner: among gated_n>=15, maximize (gated_ret, gated_win, -|maxdd|)
    cand = rdf[rdf["gated_n"].fillna(0) >= 15].copy()
    if cand.empty:
        winner = None
    else:
        cand["score"] = (
            cand["gated_ret"].fillna(0)
            + 0.5 * cand["gated_win"].fillna(0)
            + 0.2 * cand["gate_lift"].fillna(0)
            - 0.3 * cand["gated_maxdd"].fillna(0).abs()
        )
        winner = cand.sort_values("score", ascending=False).iloc[0].to_dict()

    summary = {"winner": winner, "rows": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "REPORT.md").write_text(
        "# Smooth Bouncer Bake-off\n\n```\n"
        + rdf.drop(columns=["trades"], errors="ignore").to_string(index=False)
        + "\n```\n\n"
        + (f"**Winner: `{winner['name']}`**\n" if winner else "No winner.\n")
    )
    print("\n=== BAKEOFF ===", flush=True)
    print(rdf[["name", "auc_valid", "gate_lift", "gate_precision", "ungated_ret", "gated_ret", "gated_win", "gated_n", "gated_maxdd"]].to_string(index=False), flush=True)
    if winner:
        print("WINNER:", winner["name"], flush=True)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
