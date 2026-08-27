#!/usr/bin/env python3
"""Phase 2: Top2 Entry Validator — reject obvious FA on real seats only.

Trains LightGBM on Phase-1 seats.parquet (clear_true vs clear_false).
Primary label: y_train_atr. KPI on walk-forward OOS:
  FA_removed ≥ 20%, true_signal_loss ≤ 10%, reject_precision ≥ 80%.
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

from maga7.common.decision_funnel import FUNNEL_VERSION
from maga7.common.lgbm_bouncer import save_lgbm_model
from maga7.tools.bakeoff_smooth_bouncer import USEFUL_FEATS
from maga7.tools.train_signal_validator_reject import (
    _auc,
    _eval_reject_at_thr,
    pick_operating_point,
    reject_curve,
)

MS_FEATS = (
    "ms3_ret",
    "ms3_path_eff",
    "ms3_max_dd",
    "ms5_ret",
    "ms5_path_eff",
    "ms5_max_dd",
    "ms10_ret",
    "ms10_path_eff",
    "ms10_max_dd",
    "ms20_ret",
    "ms20_path_eff",
    "ms20_max_dd",
)
EXTRA = ("atr_pct", "seat_rank", "n_day_candidates", "dir_sign", "sleeve_smooth")

# Expanding walk-forward folds (train → calib → test)
DEFAULT_FOLDS = [
    {
        "name": "fold_2025h1",
        "train_end": "2024-12-31",
        "calib_start": "2025-01-01",
        "calib_end": "2025-03-31",
        "test_start": "2025-04-01",
        "test_end": "2025-06-30",
    },
    {
        "name": "fold_2025h2",
        "train_end": "2025-06-30",
        "calib_start": "2025-07-01",
        "calib_end": "2025-09-30",
        "test_start": "2025-10-01",
        "test_end": "2025-12-31",
    },
    {
        "name": "fold_2026h1",
        "train_end": "2025-12-31",
        "calib_start": "2026-01-01",
        "calib_end": "2026-03-31",
        "test_start": "2026-04-01",
        "test_end": "2026-07-17",
    },
]


def _prep_seats(ds: pd.DataFrame, *, label: str = "atr") -> pd.DataFrame:
    out = ds.copy()
    out["date"] = out["date"].astype(str)
    out["sleeve_smooth"] = (out["sleeve"].astype(str) == "smooth").astype(float)
    if label == "atr":
        out["y_allow"] = out["y_clear_true_atr"].astype(int)
        out["y_toxic"] = out["y_clear_false_atr"].astype(int)
        out["y_train"] = out["y_train_atr"]
    else:
        out["y_allow"] = out["y_clear_true_pct"].astype(int)
        out["y_toxic"] = out["y_clear_false_pct"].astype(int)
        out["y_train"] = out["y_train_pct"]
    return out


def _feat_cols(ds: pd.DataFrame) -> list[str]:
    cols = []
    for c in list(USEFUL_FEATS) + list(MS_FEATS) + list(EXTRA):
        if c in ds.columns and c not in cols:
            cols.append(c)
    return cols


def train_fold(
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
    fa_target: float,
    seed: int,
    out: Path,
) -> dict:
    dates = ds["date"].astype(str)
    # Train on clear labels only
    clear = ds["y_train"].notna()
    tr = ds.loc[clear & (dates <= train_end)].copy()
    ca = ds.loc[clear & (dates >= calib_start) & (dates <= calib_end)].copy()
    te = ds.loc[clear & (dates >= test_start) & (dates <= test_end)].copy()
    if len(tr) < 80 or len(ca) < 20 or len(te) < 20:
        return {
            "name": name,
            "error": f"split too small train={len(tr)} calib={len(ca)} test={len(te)}",
        }

    # Train P(clear_false / toxic) — high score = reject candidate
    y_tr = tr["y_toxic"].astype(int).to_numpy()
    y_ca = ca["y_toxic"].astype(int).to_numpy()
    y_te = te["y_toxic"].astype(int).to_numpy()
    X_tr = tr[feat_cols].astype(float).fillna(0.0).to_numpy()
    X_ca = ca[feat_cols].astype(float).fillna(0.0).to_numpy()
    X_te = te[feat_cols].astype(float).fillna(0.0).to_numpy()

    # Cost-sensitive: false reject of true costs more → lower weight on toxic class
    # so model is conservative about calling toxic.
    n_pos = max(int(y_tr.sum()), 1)
    n_neg = max(int(len(y_tr) - y_tr.sum()), 1)
    # scale_pos_weight for toxic class; keep modest (< n_neg/n_pos) to avoid over-reject
    spw = min(float(n_neg / n_pos), 2.0)

    import lightgbm as lgb

    train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=list(feat_cols))
    valid_set = lgb.Dataset(X_ca, label=y_ca, feature_name=list(feat_cols), reference=train_set)
    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": 0.05,
        "num_leaves": 23,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 12,
        "lambda_l1": 0.1,
        "lambda_l2": 0.5,
        "scale_pos_weight": spw,
        "verbosity": -1,
        "seed": seed,
    }
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=400,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "calib"],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
    )

    # p_keep = 1 - P(toxic); reject when p_keep low
    p_ca = 1.0 - booster.predict(X_ca)
    p_te = 1.0 - booster.predict(X_te)
    # KPI labels: y_allow = clear_true
    y_ca_allow = ca["y_allow"].astype(int).to_numpy()
    y_te_allow = te["y_allow"].astype(int).to_numpy()

    curve_ca = reject_curve(y_ca_allow, p_ca)
    curve_te = reject_curve(y_te_allow, p_te)
    curve_ca.to_csv(out / f"reject_curve_{name}_calib.csv", index=False)
    curve_te.to_csv(out / f"reject_curve_{name}_test.csv", index=False)

    op = pick_operating_point(curve_ca, max_true_loss=max_true_loss)
    if op is None:
        return {"name": name, "error": "empty reject curve"}
    thr = float(op["thr"])
    test_op = _eval_reject_at_thr(y_te_allow, p_te, thr, max_true_loss=max_true_loss)
    # Override FA target check to architecture V2 (≥20%)
    test_op["meets_fa_target"] = bool(test_op["fa_removed"] >= fa_target)
    test_op["meets_prec"] = bool((test_op.get("prec_reject") or 0) >= 0.80)
    test_op["meets_all"] = bool(
        test_op["meets_true_loss"] and test_op["meets_fa_target"] and test_op["meets_prec"]
    )

    need = curve_te[curve_te["fa_removed"] >= fa_target].sort_values("true_lost")
    fa_cost = need.iloc[0].to_dict() if len(need) else None

    save_lgbm_model(
        booster,
        out / f"validator_{name}.txt",
        meta={
            "fold": name,
            "funnel_version": FUNNEL_VERSION,
            "feature_cols": feat_cols,
            "target": "toxic_atr",
            "score": "p_keep",
            "operating_point_calib": op,
            "operating_point_test": test_op,
            "fa_target_cost": fa_cost,
            "max_true_loss": max_true_loss,
            "fa_target": fa_target,
            "auc_keep_calib": _auc(y_ca_allow, p_ca),
            "auc_keep_test": _auc(y_te_allow, p_te),
            "n_train": int(len(tr)),
            "n_calib": int(len(ca)),
            "n_test": int(len(te)),
            "scale_pos_weight": spw,
        },
    )
    print(
        f"[{name}] train={len(tr)} calib={len(ca)} test={len(te)} | "
        f"calib FA={op['fa_removed']:.1%} lost={op['true_lost']:.1%} → "
        f"test FA={test_op['fa_removed']:.1%} lost={test_op['true_lost']:.1%} "
        f"prec_rej={test_op.get('prec_reject')} meet={test_op['meets_all']}",
        flush=True,
    )
    return {
        "name": name,
        "n_train": int(len(tr)),
        "n_calib": int(len(ca)),
        "n_test": int(len(te)),
        "auc_calib": _auc(y_ca_allow, p_ca),
        "auc_test": _auc(y_te_allow, p_te),
        "op_calib": op,
        "op_test": test_op,
        "fa_cost": fa_cost,
        "thr": thr,
        "meets_all": test_op["meets_all"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--seats",
        default="/mnt/s990/data/maga7/results/top2_decision_dataset_v1/seats.parquet",
    )
    ap.add_argument("--label", default="atr", choices=["atr", "pct"])
    ap.add_argument("--max-true-loss", type=float, default=0.10)
    ap.add_argument("--fa-target", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/top2_entry_validator_v1",
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seats_path = Path(args.seats)
    if not seats_path.exists():
        raise SystemExit(f"missing seats dataset: {seats_path}")

    ds = _prep_seats(pd.read_parquet(seats_path), label=args.label)
    feat_cols = _feat_cols(ds)
    clear_n = int(ds["y_train"].notna().sum())
    print(
        f"seats={len(ds)} clear={clear_n} allow_rate="
        f"{ds.loc[ds.y_train.notna(), 'y_allow'].mean():.3f} feats={len(feat_cols)} "
        f"label={args.label}",
        flush=True,
    )

    results = []
    for fold in DEFAULT_FOLDS:
        results.append(
            train_fold(
                fold["name"],
                ds,
                feat_cols,
                train_end=fold["train_end"],
                calib_start=fold["calib_start"],
                calib_end=fold["calib_end"],
                test_start=fold["test_start"],
                test_end=fold["test_end"],
                max_true_loss=float(args.max_true_loss),
                fa_target=float(args.fa_target),
                seed=int(args.seed),
                out=out,
            )
        )

    scored = [r for r in results if "error" not in r]
    n_meet = sum(1 for r in scored if r.get("meets_all"))
    summary = {
        "funnel_version": FUNNEL_VERSION,
        "label": args.label,
        "fa_target": args.fa_target,
        "max_true_loss": args.max_true_loss,
        "n_folds": len(scored),
        "n_folds_meet_kpi": n_meet,
        "verdict": (
            "PASS"
            if n_meet >= 3
            else ("PARTIAL" if n_meet >= 1 else "FAIL")
        ),
        "folds": [
            {
                k: v
                for k, v in r.items()
                if k not in ("booster", "p_test", "y_test")
            }
            for r in results
        ],
        "feature_cols": feat_cols,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# Top2 Entry Validator (Phase 2)",
        "",
        f"**Verdict: `{summary['verdict']}`** · folds meeting KPI: "
        f"`{n_meet}/{len(scored)}`",
        f"Label: `{args.label}` · FA≥`{args.fa_target:.0%}` · "
        f"true_loss≤`{args.max_true_loss:.0%}` · reject_prec≥80%",
        "",
    ]
    for r in results:
        if "error" in r:
            lines += [f"## {r['name']}", "", f"ERROR: {r['error']}", ""]
            continue
        t = r["op_test"]
        lines += [
            f"## {r['name']}",
            "",
            f"- n train/calib/test: `{r['n_train']}` / `{r['n_calib']}` / `{r['n_test']}`",
            f"- AUC keep calib/test: `{r['auc_calib']:.3f}` / `{r['auc_test']:.3f}`",
            f"- test FA_removed: `{t['fa_removed']:.1%}`",
            f"- test true_lost: `{t['true_lost']:.1%}`",
            f"- test prec_reject: `{t.get('prec_reject')}`",
            f"- meets_all: `{t['meets_all']}`",
            "",
        ]
    lines += [
        "## Stop rule",
        "",
        "- Need ≥3 OOS folds meeting KPI to proceed to strategy BT with replace.",
        "- If FAIL/PARTIAL, do not escalate model complexity; inspect features/labels.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({
        "verdict": summary["verdict"],
        "n_folds_meet_kpi": n_meet,
        "folds": [
            {
                "name": r.get("name"),
                "meets": r.get("meets_all"),
                "fa": (r.get("op_test") or {}).get("fa_removed"),
                "lost": (r.get("op_test") or {}).get("true_lost"),
                "error": r.get("error"),
            }
            for r in results
        ],
    }, indent=2), flush=True)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
