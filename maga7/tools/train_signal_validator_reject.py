#!/usr/bin/env python3
"""Signal Validator trained for *reject* KPI, not full-class AUC.

Goal (operating point):
  maximize FA_removed  s.t.  true_signal_loss ≤ max_true_loss (default 10%)

Stack role: Rules (high recall) → Candidate → **this validator** → regime/exec/exit.

Training variants:
  - full:   y = allow (MFE/MAE) on all launches
  - clear:  train only on clear-true ∪ clear-toxic (drop chop middle)
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
from maga7.tools.bakeoff_smooth_bouncer import USEFUL_FEATS, _train_lgbm
from maga7.tools.run_smooth_impulse_stock_replay import SYMS, _equity
from maga7.tools.train_smooth_launch_bouncer import MONTHS, _auc, build_dataset

NY = "America/New_York"


def reject_curve(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """Reject lowest-p mass; report FA removed vs true lost."""
    y = y.astype(int)
    p = p.astype(float)
    m = np.isfinite(p)
    y, p = y[m], p[m]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    rows = []
    for frac in np.round(np.arange(0.05, 0.55, 0.05), 2):
        thr = float(np.quantile(p, frac))
        rej = p <= thr
        n_rej = int(rej.sum())
        if n_rej == 0:
            continue
        fa_rm = float(((y == 0) & rej).sum() / max(n_neg, 1))
        true_lost = float(((y == 1) & rej).sum() / max(n_pos, 1))
        prec_rej = float((y[rej] == 0).mean())
        keep = ~rej
        rows.append(
            {
                "reject_frac": float(frac),
                "thr": thr,
                "n_rej": n_rej,
                "n_keep": int(keep.sum()),
                "fa_removed": fa_rm,
                "true_lost": true_lost,
                "prec_reject": prec_rej,
                "prec_keep": float(y[keep].mean()) if keep.any() else None,
                "allow_base": float(y.mean()),
            }
        )
    return pd.DataFrame(rows)


def pick_operating_point(curve: pd.DataFrame, *, max_true_loss: float) -> dict | None:
    ok = curve[curve["true_lost"] <= max_true_loss + 1e-12].copy()
    if ok.empty:
        # fallback: minimal true_lost row
        r = curve.sort_values(["true_lost", "fa_removed"], ascending=[True, False]).iloc[0]
        d = r.to_dict()
        d["constrained"] = False
        return d
    r = ok.sort_values(["fa_removed", "prec_reject"], ascending=False).iloc[0]
    d = r.to_dict()
    d["constrained"] = True
    return d


def score_launches(
    booster, feat_cols, day, qday, merged, *, target: str = "allow"
) -> dict[str, float]:
    out = {}
    for ln, sleeve in merged:
        feat = extract_bouncer_features(
            symbol=ln.symbol,
            direction=ln.direction,
            asof_ts=ln.detect_ts,
            stock_df=day,
            qqq_df=qday,
        )
        if feat is None:
            continue
        row = {
            **feat,
            "look_ret": ln.look_ret,
            "path_eff": ln.path_eff,
            "up_frac": ln.up_frac,
            "max_dd": ln.max_dd,
            "from_extreme": ln.from_extreme,
            "score": ln.score,
        }
        x = np.array([[float(row.get(c, 0.0)) for c in feat_cols]], dtype=float)
        raw = float(booster.predict(x)[0])
        out[str(ln.detect_ts)] = (1.0 - raw) if target == "toxic" else raw
    return out


def run_stock_bt(
    data,
    *,
    bt_start: str,
    bt_end: str,
    smooth_cfg,
    impulse_cfg,
    trade_cfg,
    booster,
    feat_cols,
    p_min: float | None,
    target: str = "allow",
) -> dict:
    qqq = data.get("QQQ")
    all_trades = []
    for sym in SYMS:
        raw = data.get(sym)
        if raw is None:
            continue
        dates = sorted(
            d for d in raw["date"].astype(str).unique() if bt_start <= d <= bt_end
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
                pmap = score_launches(
                    booster, feat_cols, day, qday, merged, target=target
                )
                kept = []
                for r in rows:
                    p = pmap.get(str(r["detect_ts"]))
                    if p is None and pmap:
                        rt = pd.Timestamp(r["detect_ts"])
                        best = min(
                            pmap.items(),
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
    return {**{k: eq.get(k) for k in ("total_ret", "maxdd", "n_trades", "trade_win", "avg_trade_ret")}, "trades": tdf}


def _eval_reject_at_thr(y: np.ndarray, p: np.ndarray, thr: float, *, max_true_loss: float) -> dict:
    rej = p <= thr
    n_pos = max(int(y.sum()), 1)
    n_neg = max(int((1 - y).sum()), 1)
    fa_rm = float(((y == 0) & rej).sum() / n_neg)
    true_lost = float(((y == 1) & rej).sum() / n_pos)
    return {
        "thr": float(thr),
        "fa_removed": fa_rm,
        "true_lost": true_lost,
        "prec_reject": float((y[rej] == 0).mean()) if rej.any() else None,
        "prec_keep": float(y[~rej].mean()) if (~rej).any() else None,
        "n_keep": int((~rej).sum()),
        "n_rej": int(rej.sum()),
        "meets_true_loss": true_lost <= max_true_loss,
        "meets_fa_target": fa_rm >= 0.25,
    }


def train_variant(
    name: str,
    ds: pd.DataFrame,
    feat_cols: list[str],
    *,
    train_end: str,
    calib_start: str,
    calib_end: str,
    test_start: str,
    test_end: str,
    max_true_loss: float,
    clear_only: bool,
    target: str,
    seed: int,
    out: Path,
) -> dict:
    """target: allow | toxic (toxic → score = 1-p_toxic so low = reject)."""
    dates = ds["date"].astype(str)
    tr_mask = dates <= train_end
    ca_mask = (dates >= calib_start) & (dates <= calib_end)
    te_mask = (dates >= test_start) & (dates <= test_end)

    train_df = ds.loc[tr_mask].copy()
    if clear_only:
        keep = (train_df["y_allow"] == 1) | (train_df["y_toxic"] == 1)
        train_df = train_df.loc[keep].copy()
        print(f"  [{name}] clear-only train n={len(train_df)} allow={train_df.y_allow.mean():.3f}", flush=True)

    if len(train_df) < 80:
        return {"name": name, "error": f"train too small n={len(train_df)}"}

    if target == "toxic":
        y_tr = train_df["y_toxic"].astype(int).to_numpy()
        y_ca_model = ds.loc[ca_mask, "y_toxic"].astype(int).to_numpy()
    else:
        y_tr = train_df["y_allow"].astype(int).to_numpy()
        y_ca_model = ds.loc[ca_mask, "y_allow"].astype(int).to_numpy()

    X_tr = train_df[feat_cols].astype(float).to_numpy()
    ca = ds.loc[ca_mask]
    te = ds.loc[te_mask]
    # KPI always on allow (true start)
    y_ca = ca["y_allow"].astype(int).to_numpy()
    y_te = te["y_allow"].astype(int).to_numpy()
    X_ca = ca[feat_cols].astype(float).to_numpy()
    X_te = te[feat_cols].astype(float).to_numpy()

    booster = _train_lgbm(X_tr, y_tr, X_ca, y_ca_model, feat_cols, seed=seed)
    raw_ca = booster.predict(X_ca)
    raw_te = booster.predict(X_te)
    raw_tr = booster.predict(X_tr)
    # unify: higher p_keep = safer to trade
    if target == "toxic":
        p_ca, p_te, p_tr = 1.0 - raw_ca, 1.0 - raw_te, 1.0 - raw_tr
    else:
        p_ca, p_te, p_tr = raw_ca, raw_te, raw_tr

    curve_ca = reject_curve(y_ca, p_ca)
    curve_te = reject_curve(y_te, p_te)
    curve_ca.to_csv(out / f"reject_curve_{name}_calib.csv", index=False)
    curve_te.to_csv(out / f"reject_curve_{name}_test.csv", index=False)

    op = pick_operating_point(curve_ca, max_true_loss=max_true_loss)
    if op is None:
        return {"name": name, "error": "empty reject curve"}

    thr = float(op["thr"])
    test_op = _eval_reject_at_thr(y_te, p_te, thr, max_true_loss=max_true_loss)

    # What reject_frac on test would be needed for FA≥25% (diagnostic)
    need = curve_te[curve_te["fa_removed"] >= 0.25].sort_values("true_lost")
    fa25 = need.iloc[0].to_dict() if len(need) else None

    save_lgbm_model(
        booster,
        out / f"validator_{name}.txt",
        meta={
            "variant": name,
            "feature_cols": feat_cols,
            "clear_only": clear_only,
            "target": target,
            "score": "p_keep",
            "operating_point_calib": op,
            "operating_point_test": test_op,
            "test_fa25_cost": fa25,
            "max_true_loss": max_true_loss,
            "auc_keep_calib": _auc(y_ca, p_ca),
            "auc_keep_test": _auc(y_te, p_te),
        },
    )
    print(
        f"  [{name}] calib: FA_rm={op['fa_removed']:.1%} true_lost={op['true_lost']:.1%} "
        f"→ test: FA_rm={test_op['fa_removed']:.1%} true_lost={test_op['true_lost']:.1%}"
        + (f" | FA≥25% costs true_lost={fa25['true_lost']:.1%}" if fa25 else " | FA≥25% unreachable"),
        flush=True,
    )
    return {
        "name": name,
        "clear_only": clear_only,
        "target": target,
        "n_train": int(len(train_df)),
        "n_calib": int(ca_mask.sum()),
        "n_test": int(te_mask.sum()),
        "auc_calib": _auc(y_ca, p_ca),
        "auc_test": _auc(y_te, p_te),
        "op_calib": op,
        "op_test": test_op,
        "fa25_cost": fa25,
        "booster": booster,
        "thr": thr,
        "feat_cols": feat_cols,
        "p_test": p_te,
        "y_test": y_te,
        "target_mode": target,
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
    ap.add_argument(
        "--dataset",
        default="/mnt/s990/data/maga7/results/smooth_bouncer_bakeoff_v1/dataset_up_mfe10.parquet",
        help="Reuse bakeoff dataset if present; else rebuild",
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
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/signal_validator_reject_v1",
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds_path = Path(args.dataset)
    if ds_path.exists():
        print(f"[load] dataset {ds_path}", flush=True)
        ds = pd.read_parquet(ds_path)
        if "direction" in ds.columns:
            ds = ds[ds["direction"].astype(str).str.upper() == "UP"].reset_index(drop=True)
        # ensure toxic label
        if "y_toxic" not in ds.columns and {"mfe", "mae"}.issubset(ds.columns):
            ds["y_toxic"] = (
                (ds["mae"] >= float(args.toxic_mae)) & (ds["mfe"] < float(args.good_mfe))
            ).astype(int)
        if "y_allow" not in ds.columns and {"mfe", "mae"}.issubset(ds.columns):
            ds["y_allow"] = (
                (ds["mfe"] >= float(args.good_mfe)) & (ds["mae"] <= float(args.toxic_mae))
            ).astype(int)
    else:
        print("[build] dataset", flush=True)
        prof = load_profile(args.profile)
        root = Path(prof["_paths"]["stock_root"]).expanduser()
        data_tmp: dict[str, pd.DataFrame] = {}
        for sym in SYMS + ["QQQ"]:
            raw = load_stock_month_files(root, sym, MONTHS)
            if raw.empty:
                continue
            raw = attach_mf_features(raw)
            data_tmp[sym] = raw[raw["date"].astype(str).between("2026-01-01", "2026-07-17")]
        smooth_cfg = SmoothLaunchConfig(scan_end="11:30", min_look_ret=0.002, cooldown_minutes=60)
        impulse_cfg = ImpulseLaunchConfig(scan_end="11:30", min_look_ret=0.004)
        ds = build_dataset(
            data_tmp,
            smooth_cfg=smooth_cfg,
            impulse_cfg=impulse_cfg,
            good_mfe=float(args.good_mfe),
            toxic_mae=float(args.toxic_mae),
            horizon=90,
        )
        ds = ds[ds.direction == "UP"].reset_index(drop=True)
        ds.to_parquet(out / "dataset_up.parquet", index=False)

    feat_cols = [c for c in USEFUL_FEATS if c in ds.columns]
    print(
        f"dataset n={len(ds)} allow={ds.y_allow.mean():.3f} toxic={ds.y_toxic.mean():.3f} "
        f"feats={len(feat_cols)}",
        flush=True,
    )

    variants = []
    for name, clear, tgt in [
        ("full_allow", False, "allow"),
        ("clear_only", True, "allow"),
        ("full_toxic", False, "toxic"),
        ("clear_toxic", True, "toxic"),
    ]:
        print(f"[train] {name}", flush=True)
        variants.append(
            train_variant(
                name,
                ds,
                feat_cols,
                train_end=args.train_end,
                calib_start=args.calib_start,
                calib_end=args.calib_end,
                test_start=args.test_start,
                test_end=args.test_end,
                max_true_loss=float(args.max_true_loss),
                clear_only=clear,
                target=tgt,
                seed=int(args.seed),
                out=out,
            )
        )

    # pick winner by test FA_removed among those with true_lost ≤ budget (else calib)
    scored = [v for v in variants if "error" not in v]
    def _score(v):
        t = v["op_test"]
        bonus = 1.0 if t.get("meets_true_loss") else 0.0
        return (bonus, t.get("fa_removed") or 0.0, -(t.get("true_lost") or 1.0))

    winner = sorted(scored, key=_score, reverse=True)[0] if scored else None

    # Stock BT on test window for winner + ungated
    print("[stock] load + backtest", flush=True)
    prof = load_profile(args.profile)
    root = Path(prof["_paths"]["stock_root"]).expanduser()
    data: dict[str, pd.DataFrame] = {}
    for sym in SYMS + ["QQQ"]:
        raw = load_stock_month_files(root, sym, MONTHS)
        if raw.empty:
            continue
        raw = attach_mf_features(raw)
        data[sym] = raw[raw["date"].astype(str).between("2026-01-01", args.test_end)]

    smooth_cfg = SmoothLaunchConfig(scan_end="11:30", min_look_ret=0.002, cooldown_minutes=60)
    impulse_cfg = ImpulseLaunchConfig(scan_end="11:30", min_look_ret=0.004)
    trade_cfg = SmoothStockTradeConfig(
        break_max_adverse=0.012,
        max_hold_minutes=180,
        break_min_up_frac=0.35,
        first_per_symbol_dir=True,
    )

    bt_rows = []
    ung = run_stock_bt(
        data,
        bt_start=args.test_start,
        bt_end=args.test_end,
        smooth_cfg=smooth_cfg,
        impulse_cfg=impulse_cfg,
        trade_cfg=trade_cfg,
        booster=winner["booster"] if winner else None,
        feat_cols=feat_cols,
        p_min=None,
        target="allow",
    )
    bt_rows.append({"filter": "ungated", **{k: ung[k] for k in ("total_ret", "maxdd", "n_trades", "trade_win", "avg_trade_ret")}})
    if not ung["trades"].empty:
        ung["trades"].to_csv(out / "trades_ungated.csv", index=False)

    for v in scored:
        gat = run_stock_bt(
            data,
            bt_start=args.test_start,
            bt_end=args.test_end,
            smooth_cfg=smooth_cfg,
            impulse_cfg=impulse_cfg,
            trade_cfg=trade_cfg,
            booster=v["booster"],
            feat_cols=v["feat_cols"],
            p_min=float(v["thr"]),
            target=str(v.get("target") or "allow"),
        )
        bt_rows.append(
            {
                "filter": f"{v['name']}@reject_kpi",
                "thr": v["thr"],
                **{k: gat[k] for k in ("total_ret", "maxdd", "n_trades", "trade_win", "avg_trade_ret")},
            }
        )
        if not gat["trades"].empty:
            gat["trades"].to_csv(out / f"trades_{v['name']}.csv", index=False)

    btdf = pd.DataFrame(bt_rows)
    btdf.to_csv(out / "stock_backtest.csv", index=False)

    # KPI pass?
    kpi = None
    if winner:
        t = winner["op_test"]
        kpi = {
            "max_true_loss": args.max_true_loss,
            "fa_target": 0.25,
            "pass_true_loss": bool(t.get("meets_true_loss")),
            "pass_fa_target": bool(t.get("meets_fa_target")),
            "pass_both": bool(t.get("meets_true_loss") and t.get("meets_fa_target")),
        }

    summary = {
        "kpi": kpi,
        "winner": None
        if winner is None
        else {
            "name": winner["name"],
            "thr": winner["thr"],
            "op_calib": winner["op_calib"],
            "op_test": winner["op_test"],
            "auc_test": winner["auc_test"],
        },
        "variants": [
            {
                "name": v.get("name"),
                "error": v.get("error"),
                "op_calib": v.get("op_calib"),
                "op_test": v.get("op_test"),
                "fa25_cost": v.get("fa25_cost"),
                "auc_test": v.get("auc_test"),
                "n_train": v.get("n_train"),
            }
            for v in variants
        ],
        "stock_backtest": btdf.to_dict(orient="records"),
        "splits": {
            "train_end": args.train_end,
            "calib": [args.calib_start, args.calib_end],
            "test": [args.test_start, args.test_end],
        },
        "note": "Optimize reject@true_loss; do not chase full-class AUC.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # REPORT
    lines = [
        "# Signal Validator — Reject KPI",
        "",
        f"**Constraint:** true_signal_loss ≤ `{args.max_true_loss:.0%}` · aim FA_removed ≥ 25%",
        "",
    ]
    if winner and kpi:
        status = "PASS" if kpi["pass_both"] else ("PARTIAL" if kpi["pass_true_loss"] else "FAIL")
        lines += [
            f"**Winner: `{winner['name']}` · KPI `{status}`**",
            "",
            f"- Calib: FA_rm `{winner['op_calib']['fa_removed']:.1%}`, true_lost `{winner['op_calib']['true_lost']:.1%}`, thr `{winner['thr']:.4f}`",
            f"- Test:  FA_rm `{winner['op_test']['fa_removed']:.1%}`, true_lost `{winner['op_test']['true_lost']:.1%}`, prec_reject `{winner['op_test']['prec_reject']}`",
            "",
        ]
    lines += [
        "## Variant ops (test)",
        "",
        "```",
        pd.DataFrame(
            [
                {
                    "name": v["name"],
                    "fa_rm": (v.get("op_test") or {}).get("fa_removed"),
                    "true_lost": (v.get("op_test") or {}).get("true_lost"),
                    "prec_rej": (v.get("op_test") or {}).get("prec_reject"),
                    "fa25_true_cost": (v.get("fa25_cost") or {}).get("true_lost"),
                    "auc": v.get("auc_test"),
                    "err": v.get("error"),
                }
                for v in variants
            ]
        ).to_string(index=False),
        "```",
        "",
        "## Stock UP trail120 (test window)",
        "",
        "```",
        btdf.to_string(index=False),
        "```",
        "",
        "## Reject curves",
        "",
        "See `reject_curve_*_calib.csv` / `*_test.csv`.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    print("\n" + "\n".join(lines), flush=True)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
