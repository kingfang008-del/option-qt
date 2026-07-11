#!/usr/bin/env python3
"""RightTailScore v1: supervised LightGBM on confirmed State Gate trades.

Train on Apr+May, evaluate Jun OOS.
Actionable labels (not hand-weighted ranks):
  - extend_helps: fixed_180s > ret_45s
  - peak_after_45: mfe_t > 45
  - right_tail_soft: mfe>=8% and mae>-20%
  - mfe_left: regression on mfe_left_after_45s
  - extend_edge: regression on (fixed_180s - ret_45s)

Policies gated by predicted score are compared against:
  - always_45 / always_180
  - state clock rec45_lunch180
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from factor_lab.tools.analyze_0dte_state_gate_confirm_filters import summarize as summarize_rets


FEATURE_COLS = [
    "vol_score",
    "flow_score",
    "gamma_score",
    "liquidity_score",
    "hot_score",
    "tree_edge_score",
    "time_score",
    "trend_score",
    "spread_pct",
    "quote_imbalance",
    "stock_abs_ret_60s",
    "stock_ret_60s",
    "stock_rv_60s",
    "stock_vwap_dev",
    "flow_imbalance_5s",
    "is_vol_expansion",
    "is_negative_gamma_proxy",
    "is_positive_gamma_proxy",
    "is_power_hour",
    "is_opening",
    "is_lunch",
    "is_put_trend_proxy",
    "is_call_trend_proxy",
    "is_put_flow_continuation",
    "is_put_flow_exhaustion",
    "is_high_vol_proxy",
    "is_low_vol_proxy",
    "is_recovering_state",
    "is_lunch_state",
    "is_call_side",
]


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_recovering_state"] = out["active_state"].eq("is_qqq_recovering").astype(float)
    out["is_lunch_state"] = out["active_state"].eq("is_stock_trend_down__and__is_lunch").astype(float)
    out["is_call_side"] = out["side"].eq("CALL").astype(float)
    out["extend_edge_180"] = pd.to_numeric(out["fixed_180s"], errors="coerce") - pd.to_numeric(
        out["ret_45s"], errors="coerce"
    )
    out["extend_helps_180"] = (out["extend_edge_180"] > 0).astype(float)
    out["peak_after_45"] = (pd.to_numeric(out["mfe_t"], errors="coerce") > 45).astype(float)
    out["right_tail_soft"] = out["right_tail_soft"].astype(float)
    out["mfe_left"] = pd.to_numeric(out["mfe_left_after_45s"], errors="coerce")
    return out


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FEATURE_COLS if c in df.columns]
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    return x.fillna(x.median(numeric_only=True)).fillna(0.0)


def fit_lgbm_classifier(x: pd.DataFrame, y: pd.Series, seed: int = 42) -> lgb.LGBMClassifier:
    pos = float(y.mean()) if len(y) else 0.5
    spw = (1.0 - pos) / max(pos, 1e-6)
    model = lgb.LGBMClassifier(
        n_estimators=80,
        learning_rate=0.05,
        num_leaves=7,
        min_child_samples=max(6, len(y) // 8),
        subsample=0.85,
        colsample_bytree=0.70,
        reg_alpha=0.5,
        reg_lambda=1.0,
        scale_pos_weight=min(spw, 5.0),
        random_state=seed,
        verbosity=-1,
    )
    model.fit(x, y)
    return model


def fit_lgbm_regressor(x: pd.DataFrame, y: pd.Series, seed: int = 42) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        n_estimators=80,
        learning_rate=0.05,
        num_leaves=7,
        min_child_samples=max(6, len(y) // 8),
        subsample=0.85,
        colsample_bytree=0.70,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=seed,
        verbosity=-1,
    )
    model.fit(x, y)
    return model


def fit_logit(x: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.3,
                    max_iter=500,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    pipe.fit(x, y)
    return pipe


def fit_ridge(x: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=2.0, random_state=42)),
        ]
    )
    pipe.fit(x, y)
    return pipe


def predict_score(model, x: pd.DataFrame, *, kind: str) -> np.ndarray:
    if kind == "clf":
        if hasattr(model, "predict_proba"):
            return model.predict_proba(x)[:, 1]
        return model.predict(x)
    return model.predict(x)


def day_cv_auc(train: pd.DataFrame, x_all: pd.DataFrame, y: pd.Series, *, kind: str = "clf") -> float:
    """Leave-one-day-out AUC / Spearman on train for sanity."""
    days = sorted(train["date_str"].unique())
    if len(days) < 4:
        return float("nan")
    preds = pd.Series(np.nan, index=train.index)
    for d in days:
        tr_idx = train.index[train["date_str"].ne(d)]
        te_idx = train.index[train["date_str"].eq(d)]
        if len(tr_idx) < 12 or len(te_idx) < 1:
            continue
        xtr, ytr = x_all.loc[tr_idx], y.loc[tr_idx]
        xte = x_all.loc[te_idx]
        if kind == "clf":
            if ytr.nunique() < 2:
                continue
            m = fit_lgbm_classifier(xtr, ytr)
            preds.loc[te_idx] = predict_score(m, xte, kind="clf")
        else:
            m = fit_lgbm_regressor(xtr, ytr)
            preds.loc[te_idx] = predict_score(m, xte, kind="reg")
    ok = preds.notna() & y.notna()
    if ok.sum() < 10:
        return float("nan")
    if kind == "clf":
        if y.loc[ok].nunique() < 2:
            return float("nan")
        return float(roc_auc_score(y.loc[ok], preds.loc[ok]))
    return float(pd.Series(preds.loc[ok]).corr(y.loc[ok], method="spearman"))


def state_clock_ret(g: pd.DataFrame) -> pd.Series:
    rec = g["active_state"].eq("is_qqq_recovering")
    lunch = g["active_state"].eq("is_stock_trend_down__and__is_lunch")
    out = pd.Series(np.nan, index=g.index)
    out.loc[rec] = g.loc[rec, "ret_45s"]
    out.loc[lunch] = g.loc[lunch, "fixed_180s"]
    return out


def blended(core: pd.Series, runner: pd.Series, w: float = 0.80) -> pd.Series:
    c = pd.to_numeric(core, errors="coerce")
    r = pd.to_numeric(runner, errors="coerce")
    out = w * c + (1.0 - w) * r
    miss = ~np.isfinite(r) & np.isfinite(c)
    return out.where(~miss, c)


def evaluate_policies(df: pd.DataFrame, score: pd.Series, thr: float) -> pd.DataFrame:
    high = score >= thr
    days = int(df["date_str"].nunique()) if "date_str" in df.columns else None
    policies = {
        "always_45": df["ret_45s"],
        "always_180": df["fixed_180s"],
        "state_clock_rec45_lunch180": state_clock_ret(df),
        "high_extend_180_else_45": np.where(high, df["fixed_180s"], df["ret_45s"]),
        "high_extend_120_else_45": np.where(high, df["fixed_120s"], df["ret_45s"]),
        "high_runner80_180": blended(df["ret_45s"], np.where(high, df["fixed_180s"], df["ret_45s"]), 0.80),
        "high_runner80_trail8": blended(
            df["ret_45s"],
            np.where(high, df["runner_trail8_40"], df["ret_45s"]),
            0.80,
        ),
        # state clock + only leave lunch longer when model agrees
        "state_clock_and_high": np.where(
            high,
            state_clock_ret(df),
            df["ret_45s"],
        ),
        # override: recovering always 45; lunch uses 180 only if high else 45
        "rec45_lunch180_if_high": np.where(
            df["active_state"].eq("is_qqq_recovering"),
            df["ret_45s"],
            np.where(high, df["fixed_180s"], df["ret_45s"]),
        ),
        "only_high_45": df.loc[high, "ret_45s"] if high.any() else pd.Series(dtype=float),
    }
    rows = []
    for name, rets in policies.items():
        if name == "only_high_45":
            s = pd.to_numeric(rets, errors="coerce").dropna()
            n_days = int(df.loc[high, "date_str"].nunique()) if high.any() else 0
            rows.append({"policy": name, "n_high": int(high.sum()), "high_ratio": float(high.mean()), **summarize_rets(s, name, n_days)})
        else:
            rows.append(
                {
                    "policy": name,
                    "n_high": int(high.sum()),
                    "high_ratio": float(high.mean()),
                    **summarize_rets(pd.Series(rets, index=df.index), name, days),
                }
            )
    return pd.DataFrame(rows)


def pick_threshold(train: pd.DataFrame, score: pd.Series) -> dict:
    """Choose train quantile that maximizes train avg of rec45_lunch180_if_high vs always_45."""
    base = float(train["ret_45s"].mean())
    clock = float(state_clock_ret(train).mean())
    rows = []
    for q in [0.40, 0.50, 0.60, 0.67, 0.75]:
        thr = float(score.quantile(q))
        high = score >= thr
        # primary selection objective: lunch-gated extension
        pol = np.where(
            train["active_state"].eq("is_qqq_recovering"),
            train["ret_45s"],
            np.where(high, train["fixed_180s"], train["ret_45s"]),
        )
        avg = float(pd.Series(pol).mean())
        rows.append(
            {
                "q": q,
                "thr": thr,
                "n_high": int(high.sum()),
                "avg_rec45_lunch180_if_high": avg,
                "lift_vs_45": avg - base,
                "lift_vs_clock": avg - clock,
                "avg_high_extend_180": float(np.where(high, train["fixed_180s"], train["ret_45s"]).mean()),
            }
        )
    tab = pd.DataFrame(rows).sort_values(["lift_vs_45", "avg_rec45_lunch180_if_high"], ascending=False)
    best = tab.iloc[0].to_dict()
    return {"candidates": tab.to_dict("records"), "best": best}


def importance_table(model, feature_names: list[str], kind: str) -> list[dict]:
    if kind == "lgbm_clf" or kind == "lgbm_reg":
        vals = model.feature_importances_
    elif kind == "logit":
        coef = model.named_steps["clf"].coef_.ravel()
        vals = np.abs(coef)
    elif kind == "ridge":
        coef = model.named_steps["reg"].coef_.ravel()
        vals = np.abs(coef)
    else:
        return []
    order = np.argsort(vals)[::-1]
    return [{"feature": feature_names[i], "importance": float(vals[i])} for i in order[:15]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--trades",
        default="factor_lab/results/0dte_state_gate_right_tail_apr_jun/trade_right_tail.parquet",
    )
    p.add_argument("--fit-months", default="2026-04,2026-05")
    p.add_argument("--test-month", default="2026-06")
    p.add_argument("--output-dir", default="factor_lab/results/0dte_state_gate_right_tail_v1_apr_jun")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fit_months = [m.strip() for m in args.fit_months.split(",") if m.strip()]

    raw = pd.read_parquet(args.trades)
    df = prepare_frame(raw)
    train = df[df["month"].isin(fit_months)].copy()
    test = df[df["month"].eq(args.test_month)].copy()
    if train.empty or test.empty:
        raise SystemExit(f"empty split train={len(train)} test={len(test)}")

    x_train = feature_matrix(train)
    x_test = feature_matrix(test)
    x_all = feature_matrix(df)
    # align columns
    cols = list(x_train.columns)
    x_test = x_test.reindex(columns=cols, fill_value=0.0)
    x_all = x_all.reindex(columns=cols, fill_value=0.0)

    targets = {
        "extend_helps_180": ("clf", train["extend_helps_180"], test["extend_helps_180"]),
        "peak_after_45": ("clf", train["peak_after_45"], test["peak_after_45"]),
        "right_tail_soft": ("clf", train["right_tail_soft"], test["right_tail_soft"]),
        "mfe_left": ("reg", train["mfe_left"], test["mfe_left"]),
        "extend_edge_180": ("reg", train["extend_edge_180"], test["extend_edge_180"]),
    }

    model_rows = []
    score_frames = {"train": train[["timestamp", "date_str", "month", "active_state", "side"]].copy()}
    score_frames["test"] = test[["timestamp", "date_str", "month", "active_state", "side"]].copy()
    score_frames["all"] = df[["timestamp", "date_str", "month", "active_state", "side"]].copy()

    for tname, (kind, ytr, yte) in targets.items():
        ytr = pd.to_numeric(ytr, errors="coerce")
        yte = pd.to_numeric(yte, errors="coerce")
        ok_tr = ytr.notna()
        if kind == "clf" and ytr.loc[ok_tr].nunique() < 2:
            continue

        if kind == "clf":
            lgbm = fit_lgbm_classifier(x_train.loc[ok_tr], ytr.loc[ok_tr])
            logit = fit_logit(x_train.loc[ok_tr], ytr.loc[ok_tr])
            models = {
                f"lgbm__{tname}": ("clf", "lgbm_clf", lgbm),
                f"logit__{tname}": ("clf", "logit", logit),
            }
            cv = day_cv_auc(train.loc[ok_tr], x_train.loc[ok_tr], ytr.loc[ok_tr], kind="clf")
        else:
            lgbm = fit_lgbm_regressor(x_train.loc[ok_tr], ytr.loc[ok_tr])
            ridge = fit_ridge(x_train.loc[ok_tr], ytr.loc[ok_tr])
            models = {
                f"lgbm__{tname}": ("reg", "lgbm_reg", lgbm),
                f"ridge__{tname}": ("reg", "ridge", ridge),
            }
            cv = day_cv_auc(train.loc[ok_tr], x_train.loc[ok_tr], ytr.loc[ok_tr], kind="reg")

        for sname, (pkind, ikind, model) in models.items():
            ptr = predict_score(model, x_train, kind=pkind)
            pte = predict_score(model, x_test, kind=pkind)
            pall = predict_score(model, x_all, kind=pkind)
            score_frames["train"][sname] = ptr
            score_frames["test"][sname] = pte
            score_frames["all"][sname] = pall

            # OOS rank quality
            if pkind == "clf":
                auc = float(roc_auc_score(yte, pte)) if yte.nunique() > 1 else float("nan")
                ic = float(pd.Series(pte).corr(yte, method="spearman"))
            else:
                auc = float("nan")
                ic = float(pd.Series(pte).corr(yte, method="spearman"))

            # also IC vs actionable extend edge on test
            ic_extend = float(pd.Series(pte).corr(test["extend_edge_180"], method="spearman"))
            ic_mfe_left = float(pd.Series(pte).corr(test["mfe_left"], method="spearman"))

            thr_info = pick_threshold(train, pd.Series(ptr, index=train.index))
            thr = float(thr_info["best"]["thr"])
            pol_tr = evaluate_policies(train, pd.Series(ptr, index=train.index), thr)
            pol_te = evaluate_policies(test, pd.Series(pte, index=test.index), thr)
            pol_tr.to_csv(out_dir / f"policies_train_{sname}.csv", index=False)
            pol_te.to_csv(out_dir / f"policies_jun_{sname}.csv", index=False)

            base_te = float(pol_te.loc[pol_te["policy"].eq("always_45"), "avg_return"].iloc[0])
            clock_te = float(pol_te.loc[pol_te["policy"].eq("state_clock_rec45_lunch180"), "avg_return"].iloc[0])
            # best jun policy among model-gated ones that beat always_45 on train
            merged = pol_tr.merge(pol_te, on="policy", suffixes=("_tr", "_te"))
            gated = merged[merged["policy"].str.contains("high_|if_high|only_high|state_clock_and")]
            gated = gated[gated["avg_return_tr"] >= float(pol_tr.loc[pol_tr.policy.eq("always_45"), "avg_return"].iloc[0]) - 1e-12]
            gated = gated.sort_values(["avg_return_te", "total_return_10pct_position_te"], ascending=False)

            model_rows.append(
                {
                    "score": sname,
                    "target": tname,
                    "model": ikind,
                    "train_cv": cv,
                    "jun_auc": auc,
                    "jun_ic_target": ic,
                    "jun_ic_extend_edge": ic_extend,
                    "jun_ic_mfe_left": ic_mfe_left,
                    "thr_q": thr_info["best"]["q"],
                    "thr": thr,
                    "train_base_45": float(pol_tr.loc[pol_tr.policy.eq("always_45"), "avg_return"].iloc[0]),
                    "train_clock": float(pol_tr.loc[pol_tr.policy.eq("state_clock_rec45_lunch180"), "avg_return"].iloc[0]),
                    "jun_base_45": base_te,
                    "jun_clock": clock_te,
                    "jun_best_gated_policy": gated.iloc[0]["policy"] if not gated.empty else None,
                    "jun_best_gated_avg": float(gated.iloc[0]["avg_return_te"]) if not gated.empty else float("nan"),
                    "jun_best_gated_beats_45": bool(gated.iloc[0]["avg_return_te"] > base_te) if not gated.empty else False,
                    "jun_best_gated_beats_clock": bool(gated.iloc[0]["avg_return_te"] > clock_te) if not gated.empty else False,
                    "top_features": importance_table(model, cols, ikind),
                    "threshold_search": thr_info,
                }
            )

    models_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("top_features", "threshold_search")} for r in model_rows])
    models_df = models_df.sort_values(
        ["jun_best_gated_beats_clock", "jun_best_gated_avg", "jun_ic_extend_edge"],
        ascending=False,
    )
    models_df.to_csv(out_dir / "model_leaderboard.csv", index=False)

    # attach best scores onto full diag for inspection
    scored = df.copy()
    for col in score_frames["all"].columns:
        if col.startswith("lgbm__") or col.startswith("logit__") or col.startswith("ridge__"):
            scored[col] = score_frames["all"][col].to_numpy()
    scored.to_parquet(out_dir / "trade_right_tail_v1_scored.parquet", index=False)

    # recommendation
    usable = sorted(
        [r for r in model_rows if r.get("jun_best_gated_beats_45")],
        key=lambda r: (r.get("jun_best_gated_avg") or -1e9, r.get("jun_ic_extend_edge") or -1e9),
        reverse=True,
    )
    beats_clock = [r for r in usable if r.get("jun_best_gated_beats_clock")]
    recommendation = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "state_clock_jun_avg": float(state_clock_ret(test).mean()),
        "always_45_jun_avg": float(test["ret_45s"].mean()),
        "any_model_beats_45_on_jun": bool(usable),
        "any_model_beats_state_clock_on_jun": bool(beats_clock),
        "best_vs_45": usable[0] if usable else None,
        "best_vs_clock": beats_clock[0] if beats_clock else None,
        "verdict": (
            "Adopt model-gated hold/runner"
            if beats_clock
            else (
                "Model helps vs flat 45s but does not beat state clock; keep state clock as default"
                if usable
                else "RightTailScore v1 does not beat always_45 on Jun; do not enable score-gated runners"
            )
        ),
    }
    # strip huge nested for top-level print friendliness
    for key in ("best_vs_45", "best_vs_clock"):
        if recommendation[key] is not None:
            recommendation[key] = {
                k: recommendation[key][k]
                for k in recommendation[key]
                if k not in ("top_features", "threshold_search")
            }
            recommendation[key]["top_features"] = model_rows[
                next(i for i, r in enumerate(model_rows) if r["score"] == recommendation[key]["score"])
            ]["top_features"][:8]

    summary = {
        "config": vars(args),
        "label_rates_train": {
            "extend_helps_180": float(train["extend_helps_180"].mean()),
            "peak_after_45": float(train["peak_after_45"].mean()),
            "right_tail_soft": float(train["right_tail_soft"].mean()),
        },
        "label_rates_jun": {
            "extend_helps_180": float(test["extend_helps_180"].mean()),
            "peak_after_45": float(test["peak_after_45"].mean()),
            "right_tail_soft": float(test["right_tail_soft"].mean()),
        },
        "leaderboard": models_df.to_dict("records"),
        "models_detail": model_rows,
        "recommendation": recommendation,
        "files": {
            "leaderboard": str(out_dir / "model_leaderboard.csv"),
            "scored_trades": str(out_dir / "trade_right_tail_v1_scored.parquet"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(
        json.dumps(
            {
                "label_rates_train": summary["label_rates_train"],
                "label_rates_jun": summary["label_rates_jun"],
                "leaderboard_top": models_df.head(10).to_dict("records"),
                "recommendation": recommendation,
            },
            indent=2,
            default=str,
        )
    )
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
