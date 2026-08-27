#!/usr/bin/env python3
"""Bake-off next Validator ideas under the same reject KPI.

Variants tried one-by-one:
  A0  baseline_full       — entry feats, y=allow
  A1  baseline_clear      — entry feats, clear true∪toxic only
  B1  strat_easy_fa       — train allow vs *easy* FA only (drop hard FA)
  B2  strat_rule_reject   — no ML; reject if easy_fa rule fires
  C1  confirm_5m          — feats at t+5m, label from confirm ts
  C2  confirm_10m         — feats at t+10m
  C3  entry_plus_conf5    — entry + confirm5 feats
  D1  two_stage           — reject worst 10% entry, then confirm5 on survivors
  D2  rule10_plus_conf5   — rule easy reject ∪ confirm5 ML reject

KPI: maximize FA_removed s.t. true_lost ≤ 10% (test 06-16→07-17; calib 05-01→06-15).
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
from maga7.common.lgbm_bouncer import extract_bouncer_features
from maga7.common.signals import attach_mf_features, load_stock_month_files
from maga7.common.smooth_trend import _prepare_day, _window_stats
from maga7.tools.bakeoff_smooth_bouncer import USEFUL_FEATS, _train_lgbm
from maga7.tools.run_smooth_impulse_stock_replay import SYMS, _equity
from maga7.tools.train_smooth_launch_bouncer import MONTHS, _auc, _fwd_mfe_mae
from maga7.tools.train_signal_validator_reject import pick_operating_point, reject_curve

NY = "America/New_York"
ENTRY_FEATS = list(USEFUL_FEATS)
CONF_FEATS = (
    "conf_ret",
    "conf_path_eff",
    "conf_up_frac",
    "conf_max_dd",
    "conf_broke",
    "conf_giveback",
    "conf_mfe",
    "conf_mae",
)


def _to_ny(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize(NY)
    return t.tz_convert(NY)


def easy_fa_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Entry-time *easy* FA heuristics (no lookahead)."""
    out = df.copy()
    tod = pd.to_numeric(out.get("tod_min"), errors="coerce").fillna(0)
    look = pd.to_numeric(out["look_ret"], errors="coerce")
    pe = pd.to_numeric(out["path_eff"], errors="coerce")
    mdd = pd.to_numeric(out["max_dd"], errors="coerce")
    fe = pd.to_numeric(out["from_extreme"], errors="coerce")
    qfp = pd.to_numeric(out.get("qqq_from_prev"), errors="coerce").fillna(0)
    out["easy_late"] = (tod >= 11 * 60).astype(int)
    out["easy_chase"] = ((look >= 0.006) & (pe < 0.45)).astype(int)
    out["easy_struct"] = ((mdd < -0.0035) | ((fe < 0.0015) & (look > 0.003))).astype(int)
    out["easy_qqq_fade"] = ((qfp < -0.001) & (look > 0.004)).astype(int)
    out["easy_fa"] = (
        (out["easy_late"] + out["easy_chase"] + out["easy_struct"] + out["easy_qqq_fade"]) > 0
    ).astype(int)
    return out


def confirm_at(
    day: pd.DataFrame,
    *,
    entry_ts,
    direction: str,
    delay_min: int,
    good_mfe: float,
    toxic_mae: float,
    horizon: int,
) -> dict | None:
    day = _prepare_day(day, str(_to_ny(entry_ts).strftime("%Y-%m-%d")))
    et = _to_ny(entry_ts)
    ct = et + pd.Timedelta(minutes=delay_min)
    win = day[(day["timestamp"] >= et) & (day["timestamp"] <= ct)]
    if len(win) < max(3, delay_min // 2):
        return None
    closes = win["close"].astype(float).to_numpy()
    st = _window_stats(closes, direction=direction)
    if st is None:
        return None
    px0 = float(closes[0])
    px1 = float(closes[-1])
    if direction == "UP":
        peak = float(np.max(closes))
        giveback = (peak / px1 - 1.0) if px1 > 0 else 0.0
        broke = 1.0 if px1 < px0 else 0.0
        # path mfe/mae in confirm window
        if "high" in win.columns:
            mfe = float(pd.to_numeric(win["high"], errors="coerce").max() / px0 - 1.0)
            mae = float(-(pd.to_numeric(win["low"], errors="coerce").min() / px0 - 1.0))
        else:
            mfe = float(closes.max() / px0 - 1.0)
            mae = float(-(closes.min() / px0 - 1.0))
    else:
        trough = float(np.min(closes))
        giveback = (px1 / trough - 1.0) if trough > 0 else 0.0
        broke = 1.0 if px1 > px0 else 0.0
        if "high" in win.columns:
            mfe = float(1.0 - pd.to_numeric(win["low"], errors="coerce").min() / px0)
            mae = float(pd.to_numeric(win["high"], errors="coerce").max() / px0 - 1.0)
        else:
            mfe = float(1.0 - closes.min() / px0)
            mae = float(closes.max() / px0 - 1.0)
    # forward label from confirm time
    fmfe, fmae = _fwd_mfe_mae(day, entry_ts=ct, direction=direction, horizon_minutes=horizon)
    if fmfe is None:
        return None
    return {
        "confirm_ts": str(ct),
        "conf_ret": st["look_ret"],
        "conf_path_eff": st["path_eff"],
        "conf_up_frac": st["up_frac"],
        "conf_max_dd": st["max_dd"],
        "conf_broke": broke,
        "conf_giveback": float(max(0.0, giveback)),
        "conf_mfe": float(mfe),
        "conf_mae": float(max(0.0, mae)),
        "y_allow_conf": int(fmfe >= good_mfe and fmae <= toxic_mae),
        "mfe_conf": fmfe,
        "mae_conf": fmae,
    }


def enrich_confirms(
    ds: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    *,
    delays: tuple[int, ...] = (5, 10),
    good_mfe: float,
    toxic_mae: float,
    horizon: int = 90,
) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(ds.itertuples(index=False)):
        if i % 400 == 0:
            print(f"  [confirm] {i}/{len(ds)}", flush=True)
        sym = str(r.symbol)
        date = str(r.date)
        raw = data.get(sym)
        if raw is None:
            continue
        day = raw[raw["date"].astype(str) == date]
        if day.empty:
            continue
        base = r._asdict() if hasattr(r, "_asdict") else dict(zip(ds.columns, r))
        ok = True
        for d in delays:
            c = confirm_at(
                day,
                entry_ts=base["detect_ts"],
                direction=str(base["direction"]),
                delay_min=d,
                good_mfe=good_mfe,
                toxic_mae=toxic_mae,
                horizon=horizon,
            )
            if c is None:
                ok = False
                break
            for k, v in c.items():
                base[f"{k}_{d}" if not k.startswith("y_") and k not in ("confirm_ts",) else k] = v
            # store delay-specific y and conf feats with suffix
            base[f"y_allow_conf_{d}"] = c["y_allow_conf"]
            for fk in CONF_FEATS:
                base[f"{fk}_{d}"] = c[fk]
            base[f"confirm_ts_{d}"] = c["confirm_ts"]
        if ok:
            rows.append(base)
    return pd.DataFrame(rows)


def _split_masks(dates: pd.Series, train_end, calib_start, calib_end, test_start, test_end):
    d = dates.astype(str)
    return (
        d <= train_end,
        (d >= calib_start) & (d <= calib_end),
        (d >= test_start) & (d <= test_end),
    )


def eval_scores(
    name: str,
    y_ca: np.ndarray,
    p_ca: np.ndarray,
    y_te: np.ndarray,
    p_te: np.ndarray,
    *,
    max_true_loss: float,
) -> dict:
    curve_ca = reject_curve(y_ca, p_ca)
    curve_te = reject_curve(y_te, p_te)
    op = pick_operating_point(curve_ca, max_true_loss=max_true_loss)
    if op is None:
        return {"name": name, "error": "empty curve"}
    thr = float(op["thr"])
    rej = p_te <= thr
    n_pos = max(int(y_te.sum()), 1)
    n_neg = max(int((1 - y_te).sum()), 1)
    fa_rm = float(((y_te == 0) & rej).sum() / n_neg)
    true_lost = float(((y_te == 1) & rej).sum() / n_pos)
    need = curve_te[curve_te["fa_removed"] >= 0.25]
    fa25 = float(need["true_lost"].min()) if len(need) else None
    return {
        "name": name,
        "thr": thr,
        "fa_rm_calib": float(op["fa_removed"]),
        "true_lost_calib": float(op["true_lost"]),
        "fa_rm_test": fa_rm,
        "true_lost_test": true_lost,
        "prec_reject_test": float((y_te[rej] == 0).mean()) if rej.any() else None,
        "prec_keep_test": float(y_te[~rej].mean()) if (~rej).any() else None,
        "auc_test": _auc(y_te, p_te),
        "fa25_true_cost": fa25,
        "n_test": int(len(y_te)),
        "n_keep_test": int((~rej).sum()),
        "pass_true": true_lost <= max_true_loss,
        "pass_fa25": fa_rm >= 0.25,
        "p_test": p_te,
        "y_test": y_te,
        "curve_te": curve_te,
    }


def train_lgbm_keep(
    ds: pd.DataFrame,
    feat_cols: list[str],
    *,
    y_col: str,
    train_mask: pd.Series,
    calib_mask: pd.Series,
    train_filter: pd.Series | None,
    seed: int,
):
    tr = ds.loc[train_mask].copy()
    if train_filter is not None:
        tr = tr.loc[train_filter.loc[tr.index]].copy()
    ca = ds.loc[calib_mask]
    y_tr = tr[y_col].astype(int).to_numpy()
    y_ca = ca[y_col].astype(int).to_numpy()
    X_tr = tr[feat_cols].astype(float).to_numpy()
    X_ca = ca[feat_cols].astype(float).to_numpy()
    if len(tr) < 80 or len(np.unique(y_tr)) < 2:
        return None, None, None
    booster = _train_lgbm(X_tr, y_tr, X_ca, y_ca, feat_cols, seed=seed)
    return booster, feat_cols, tr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        default="/mnt/s990/data/maga7/results/smooth_bouncer_bakeoff_v1/dataset_up_mfe10.parquet",
    )
    ap.add_argument(
        "--profile",
        default=(
            "maga7/CONFIG/strategy_profiles/"
            "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
        ),
    )
    ap.add_argument("--train-end", default="2026-04-30")
    ap.add_argument("--calib-start", default="2026-05-01")
    ap.add_argument("--calib-end", default="2026-06-15")
    ap.add_argument("--test-start", default="2026-06-16")
    ap.add_argument("--test-end", default="2026-07-17")
    ap.add_argument("--max-true-loss", type=float, default=0.10)
    ap.add_argument("--good-mfe", type=float, default=0.010)
    ap.add_argument("--toxic-mae", type=float, default=0.008)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="/mnt/s990/data/maga7/results/validator_next_bakeoff_v1")
    ap.add_argument("--skip-confirm-rebuild", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = pd.read_parquet(args.dataset)
    ds = ds[ds.direction.astype(str).str.upper() == "UP"].reset_index(drop=True)
    ds = easy_fa_flags(ds)

    print("[load] stocks Jan–Jul", flush=True)
    prof = load_profile(args.profile)
    root = Path(prof["_paths"]["stock_root"]).expanduser()
    data: dict[str, pd.DataFrame] = {}
    for sym in SYMS + ["QQQ"]:
        raw = load_stock_month_files(root, sym, MONTHS)
        if raw.empty:
            continue
        data[sym] = attach_mf_features(raw)
        print(f"  {sym}: {raw['date'].min()}→{raw['date'].max()} n={len(raw)}", flush=True)

    results = []

    def _run_ml(name, frame, feats, y_col, train_filter_fn=None):
        feats = [c for c in feats if c in frame.columns]
        tr_mask, ca_mask, te_mask = _split_masks(
            frame["date"],
            args.train_end,
            args.calib_start,
            args.calib_end,
            args.test_start,
            args.test_end,
        )
        tfilter = train_filter_fn(frame) if train_filter_fn is not None else None
        n_tr = int(tr_mask.sum() if tfilter is None else (tr_mask & tfilter).sum())
        booster, feats, _ = train_lgbm_keep(
            frame,
            feats,
            y_col=y_col,
            train_mask=tr_mask,
            calib_mask=ca_mask,
            train_filter=tfilter,
            seed=int(args.seed),
        )
        if booster is None:
            return {"name": name, "error": f"train failed n_tr={n_tr}"}
        ca = frame.loc[ca_mask]
        te = frame.loc[te_mask]
        p_ca = booster.predict(ca[feats].astype(float).to_numpy())
        p_te = booster.predict(te[feats].astype(float).to_numpy())
        y_ca = ca[y_col].astype(int).to_numpy()
        y_te = te[y_col].astype(int).to_numpy()
        r = eval_scores(name, y_ca, p_ca, y_te, p_te, max_true_loss=float(args.max_true_loss))
        r["n_train"] = n_tr
        r["curve_te"].to_csv(out / f"curve_{name}.csv", index=False)
        print(
            f"  [{name}] FA_rm={r['fa_rm_test']:.1%} true_lost={r['true_lost_test']:.1%} "
            f"fa25_cost={r['fa25_true_cost']}",
            flush=True,
        )
        r["booster"] = booster
        r["feats"] = feats
        return r

    # --- A/B on full entry dataset ---
    print("[A0] baseline_full", flush=True)
    results.append(_run_ml("A0_baseline_full", ds, ENTRY_FEATS, "y_allow"))

    print("[A1] baseline_clear", flush=True)
    results.append(
        _run_ml(
            "A1_baseline_clear",
            ds,
            ENTRY_FEATS,
            "y_allow",
            train_filter_fn=lambda f: (f["y_allow"] == 1) | (f["y_toxic"] == 1),
        )
    )

    print("[B1] strat_easy_fa", flush=True)
    results.append(
        _run_ml(
            "B1_strat_easy_fa",
            ds,
            ENTRY_FEATS,
            "y_allow",
            train_filter_fn=lambda f: (f["y_allow"] == 1) | ((f["y_allow"] == 0) & (f["easy_fa"] == 1)),
        )
    )

    print("[B2] strat_rule_reject", flush=True)
    _, ca_m_ds, te_m_ds = _split_masks(
        ds["date"], args.train_end, args.calib_start, args.calib_end, args.test_start, args.test_end
    )
    ca = ds.loc[ca_m_ds]
    te = ds.loc[te_m_ds]
    # score: 1 if not easy_fa else 0  (reject easy)
    # soft score: -easy flags count
    p_ca = 1.0 - ca["easy_fa"].astype(float).to_numpy()
    p_te = 1.0 - te["easy_fa"].astype(float).to_numpy()
    # also continuous: 1 - 0.25*sum flags
    flag_sum_ca = ca[["easy_late", "easy_chase", "easy_struct", "easy_qqq_fade"]].sum(axis=1).to_numpy()
    flag_sum_te = te[["easy_late", "easy_chase", "easy_struct", "easy_qqq_fade"]].sum(axis=1).to_numpy()
    p_ca = 1.0 - 0.25 * flag_sum_ca
    p_te = 1.0 - 0.25 * flag_sum_te
    r = eval_scores(
        "B2_strat_rule",
        ca["y_allow"].to_numpy(int),
        p_ca,
        te["y_allow"].to_numpy(int),
        p_te,
        max_true_loss=float(args.max_true_loss),
    )
    # hard rule diagnostics: reject all easy_fa
    for split_name, frame in [("calib", ca), ("test", te)]:
        rej = frame["easy_fa"] == 1
        y = frame["y_allow"].to_numpy(int)
        n_pos, n_neg = max(y.sum(), 1), max((1 - y).sum(), 1)
        r[f"hard_rule_{split_name}"] = {
            "fa_rm": float(((y == 0) & rej).sum() / n_neg),
            "true_lost": float(((y == 1) & rej).sum() / n_pos),
            "n_rej": int(rej.sum()),
        }
    results.append(r)
    print(
        f"  [B2] FA_rm={r['fa_rm_test']:.1%} true_lost={r['true_lost_test']:.1%} "
        f"hard_test={r['hard_rule_test']}",
        flush=True,
    )

    # --- confirm enrichment (full Jan–Jul stocks) ---
    conf_path = out / "dataset_with_confirm.parquet"
    if args.skip_confirm_rebuild and conf_path.exists():
        print(f"[load] {conf_path}", flush=True)
        dsc = pd.read_parquet(conf_path)
    else:
        print("[enrich] confirm 5m/10m", flush=True)
        dsc = enrich_confirms(
            ds,
            data,
            delays=(5, 10),
            good_mfe=float(args.good_mfe),
            toxic_mae=float(args.toxic_mae),
        )
        dsc = easy_fa_flags(dsc)
        dsc.to_parquet(conf_path, index=False)
        print(f"  confirm rows {len(dsc)} / {len(ds)}", flush=True)

    tr_m, ca_m, te_m = _split_masks(
        dsc["date"],
        args.train_end,
        args.calib_start,
        args.calib_end,
        args.test_start,
        args.test_end,
    )
    ca_c = dsc.loc[ca_m]
    te_c = dsc.loc[te_m]

    for delay, tag in [(5, "C1_confirm_5m"), (10, "C2_confirm_10m")]:
        print(f"[{tag}]", flush=True)
        feats = [f"{c}_{delay}" for c in CONF_FEATS] + [
            c for c in ("tod_min", "look_ret", "path_eff", "sleeve_smooth") if c in dsc.columns
        ]
        results.append(_run_ml(tag, dsc, feats, f"y_allow_conf_{delay}"))

    print("[C3] entry_plus_conf5", flush=True)
    results.append(
        _run_ml(
            "C3_entry_plus_conf5",
            dsc,
            ENTRY_FEATS + [f"{c}_5" for c in CONF_FEATS],
            "y_allow_conf_5",
        )
    )

    print("[D1] two_stage", flush=True)
    feats_e = [c for c in ENTRY_FEATS if c in dsc.columns]
    booster_e, _, _ = train_lgbm_keep(
        dsc, feats_e, y_col="y_allow", train_mask=tr_m, calib_mask=ca_m, train_filter=None, seed=int(args.seed)
    )
    feats_c = [f"{c}_5" for c in CONF_FEATS]
    booster_c, feats_c, _ = train_lgbm_keep(
        dsc,
        feats_c,
        y_col="y_allow_conf_5",
        train_mask=tr_m,
        calib_mask=ca_m,
        train_filter=None,
        seed=int(args.seed) + 1,
    )
    if booster_e is None or booster_c is None:
        results.append({"name": "D1_two_stage", "error": "train failed"})
    else:
        pe_ca = booster_e.predict(ca_c[feats_e].astype(float).to_numpy())
        pc_ca = booster_c.predict(ca_c[feats_c].astype(float).to_numpy())
        pe_te = booster_e.predict(te_c[feats_e].astype(float).to_numpy())
        pc_te = booster_c.predict(te_c[feats_c].astype(float).to_numpy())
        thr1 = float(np.quantile(pe_ca, 0.10))

        def combine(pe, pc, thr):
            p = pc.copy()
            p[pe <= thr] = -1.0
            return p

        r = eval_scores(
            "D1_two_stage",
            ca_c["y_allow_conf_5"].to_numpy(int),
            combine(pe_ca, pc_ca, thr1),
            te_c["y_allow_conf_5"].to_numpy(int),
            combine(pe_te, pc_te, thr1),
            max_true_loss=float(args.max_true_loss),
        )
        r["stage1_thr"] = thr1
        r["stage1_reject_frac_test"] = float((pe_te <= thr1).mean())
        results.append(r)
        print(
            f"  [D1] FA_rm={r['fa_rm_test']:.1%} true_lost={r['true_lost_test']:.1%} "
            f"stage1_frac={r['stage1_reject_frac_test']:.1%}",
            flush=True,
        )

    print("[D2] rule_plus_conf5", flush=True)
    if booster_c is not None:
        pc_ca = booster_c.predict(ca_c[feats_c].astype(float).to_numpy())
        pc_te = booster_c.predict(te_c[feats_c].astype(float).to_numpy())
        r = eval_scores(
            "D2_rule_plus_conf5",
            ca_c["y_allow_conf_5"].to_numpy(int),
            np.where(ca_c["easy_fa"].to_numpy() == 1, -1.0, pc_ca),
            te_c["y_allow_conf_5"].to_numpy(int),
            np.where(te_c["easy_fa"].to_numpy() == 1, -1.0, pc_te),
            max_true_loss=float(args.max_true_loss),
        )
        results.append(r)
        print(f"  [D2] FA_rm={r['fa_rm_test']:.1%} true_lost={r['true_lost_test']:.1%}", flush=True)
    else:
        results.append({"name": "D2_rule_plus_conf5", "error": "no confirm model"})

    # Scoreboard (drop bulky arrays / models)
    drop_keys = {"p_test", "y_test", "curve_te", "booster", "feats"}
    rows = []
    for r in results:
        rows.append({k: v for k, v in r.items() if k not in drop_keys})
    sdf = pd.DataFrame(rows)
    sdf.to_csv(out / "scoreboard.csv", index=False)

    # ranking: among pass_true, max fa_rm
    if "pass_true" in sdf.columns and sdf["pass_true"].fillna(False).any():
        ok = sdf[sdf["pass_true"] == True]
        winner = ok.sort_values(["fa_rm_test", "prec_reject_test"], ascending=False).iloc[0]
    else:
        winner = sdf.dropna(subset=["fa_rm_test"]).sort_values("fa_rm_test", ascending=False).iloc[0]

    # easy_fa coverage diagnostic (entry test set)
    te_all = ds.loc[te_m_ds]
    diag = {
        "test_n": int(len(te_all)),
        "test_allow_rate": float(te_all.y_allow.mean()),
        "easy_fa_rate": float(te_all.easy_fa.mean()),
        "easy_fa_among_FA": float(te_all.loc[te_all.y_allow == 0, "easy_fa"].mean()),
        "easy_fa_among_true": float(te_all.loc[te_all.y_allow == 1, "easy_fa"].mean()),
        "easy_flag_rates": {
            c: float(te_all[c].mean())
            for c in ["easy_late", "easy_chase", "easy_struct", "easy_qqq_fade"]
        },
    }

    summary = {
        "winner": winner.to_dict(),
        "kpi": {
            "max_true_loss": args.max_true_loss,
            "fa_target": 0.25,
            "winner_pass_both": bool(winner.get("pass_true") and winner.get("pass_fa25")),
        },
        "easy_fa_diag": diag,
        "scoreboard": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    show = sdf[
        [
            c
            for c in [
                "name",
                "fa_rm_test",
                "true_lost_test",
                "prec_reject_test",
                "fa25_true_cost",
                "auc_test",
                "pass_true",
                "pass_fa25",
                "error",
            ]
            if c in sdf.columns
        ]
    ].sort_values("fa_rm_test", ascending=False)
    report = [
        "# Validator Next Bake-off",
        "",
        f"**KPI:** FA_rm ≥ 25% @ true_lost ≤ {args.max_true_loss:.0%}",
        f"**Winner under budget:** `{winner.get('name')}` "
        f"(FA_rm={winner.get('fa_rm_test')}, true_lost={winner.get('true_lost_test')})",
        f"**Pass both:** `{summary['kpi']['winner_pass_both']}`",
        "",
        "## Easy-FA coverage (test)",
        "",
        "```",
        json.dumps(diag, indent=2),
        "```",
        "",
        "## Scoreboard (test)",
        "",
        "```",
        show.to_string(index=False),
        "```",
        "",
        "## Notes",
        "",
        "- A*: entry-only baselines",
        "- B*: easy-FA stratification / rules",
        "- C*: post-launch confirm features (label from confirm ts)",
        "- D*: two-stage combinations",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(report))
    print("\n=== SCOREBOARD ===", flush=True)
    print(show.to_string(index=False), flush=True)
    print("WINNER:", winner.get("name"), flush=True)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
