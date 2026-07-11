#!/usr/bin/env python3
"""Full validation: onset gates (not chase) on 0DTE curated + 1DTE coverage/route hooks.

Causal protocol:
  - Fingerprints already on trades_with_fingerprints.parquet (pre-entry).
  - Thresholds for persist/session fit on months < eval month (expanding),
    or fixed split: fit Jan–Apr, shadow May–Jun, forward Jul.
  - Never tune on Jul.

Gates:
  - block_high_persist: drop persist >= q_fit
  - ignition: persist < q_fit AND accel > 0
  - block_chase_late: drop (persist >= q_fit AND session_minute >= 180)
  - block_thin_high_persist: drop (thin high & persist high) using fit terciles
  - onset_combo: ignition OR (not high_persist and session < 180)
  - regime block_pred_1 (optional join)

Also reports 1DTE micro readiness and which 0DTE weak days are 1DTE-route candidates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--fingerprints",
        default="factor_lab/results/0dte_institutional_fingerprints/trades_with_fingerprints.parquet",
    )
    p.add_argument(
        "--regime-preds",
        default="factor_lab/results/0dte_qqq_deep_anchor_scaffold/causal_regime_predictions.parquet",
    )
    p.add_argument(
        "--dte1-map",
        default=str(Path.home() / "train_data/locked_targets_map_1dte_api_ladder.parquet"),
    )
    p.add_argument(
        "--dte1-micro",
        default="/mnt/s990/data/microstructure/qqq_1dte_api_ladder",
    )
    p.add_argument("--position-frac", type=float, default=0.25)
    p.add_argument(
        "--output-dir",
        default="factor_lab/results/0dte_onset_full_validation",
    )
    return p.parse_args()


def account(returns: pd.Series, pf: float) -> dict:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return {"n": 0, "avg_ret": float("nan"), "win_rate": float("nan"), "account_ret": 0.0, "max_dd": 0.0}
    eq = np.cumprod(1.0 + pf * r.to_numpy())
    peaks = np.maximum.accumulate(np.r_[1.0, eq])[:-1]
    return {
        "n": int(len(r)),
        "avg_ret": float(r.mean()),
        "win_rate": float((r > 0).mean()),
        "account_ret": float(eq[-1] - 1.0),
        "max_dd": float((eq / peaks - 1.0).min()),
    }


def fit_thresholds(train: pd.DataFrame) -> dict:
    persist = pd.to_numeric(train["f3_flow_persist"], errors="coerce").dropna()
    thin = pd.to_numeric(train["f1_quote_thinning"], errors="coerce").dropna()
    return {
        "persist_q67": float(persist.quantile(0.67)) if len(persist) else 0.67,
        "persist_q50": float(persist.quantile(0.50)) if len(persist) else 0.5,
        "thin_q67": float(thin.quantile(0.67)) if len(thin) else 1.0,
        "session_late": 180.0,
    }


def apply_gates(df: pd.DataFrame, thr: dict) -> dict[str, pd.Series]:
    persist = pd.to_numeric(df["f3_flow_persist"], errors="coerce")
    accel = pd.to_numeric(df["f3_flow_accel"], errors="coerce")
    sess = pd.to_numeric(df["f3_session_minute"], errors="coerce")
    thin = pd.to_numeric(df["f1_quote_thinning"], errors="coerce")
    high_p = persist >= thr["persist_q67"]
    mid_or_low_p = persist < thr["persist_q67"]
    ignition = mid_or_low_p & (accel > 0)
    chase_late = high_p & (sess >= thr["session_late"])
    thin_chase = (thin >= thr["thin_q67"]) & high_p
    return {
        "baseline": pd.Series(True, index=df.index),
        "block_high_persist": ~high_p,
        "ignition_accel": ignition,
        "block_chase_late": ~chase_late,
        "block_thin_high_persist": ~thin_chase,
        "onset_prefer": ignition | ((~high_p) & (sess < thr["session_late"])),
        # stricter: ignition only
        "ignition_only": ignition,
    }


def eval_window(df: pd.DataFrame, thr: dict, pf: float, label: str) -> list[dict]:
    gates = apply_gates(df, thr)
    rows = []
    for name, mask in gates.items():
        stats = account(df.loc[mask, "path_exec_ret"], pf)
        stats.update(
            {
                "window": label,
                "gate": name,
                "n_keep_ratio": stats["n"] / max(len(df), 1),
                "thr_persist_q67": thr["persist_q67"],
            }
        )
        rows.append(stats)
    return rows


def dte1_coverage(map_path: Path, micro_root: Path) -> dict:
    m = pd.read_parquet(map_path)
    need = set(m["date_str"].astype(str))
    cdir = micro_root / "contract_1s" / "QQQ"
    fdir = micro_root / "features_1s" / "QQQ"
    have_c = {p.stem.replace("QQQ_", "") for p in cdir.glob("*.parquet")} if cdir.exists() else set()
    have_f = {p.stem.replace("QQQ_", "") for p in fdir.glob("*.parquet")} if fdir.exists() else set()
    # sample integrity
    sample_miss = 0
    checked = 0
    for d in sorted(need & have_c)[:: max(1, len(need) // 10)]:
        raw = pd.read_parquet(cdir / f"QQQ_{d}.parquet", columns=["ticker"])
        need_c = set(m.loc[m["date_str"] == d, "contract_symbol"].astype(str).str.replace("O:", "", regex=False))
        have = set(raw["ticker"].astype(str).str.replace("O:", "", regex=False))
        sample_miss += len(need_c - have)
        checked += 1
    return {
        "map_days": len(need),
        "contract_days": len(have_c & need),
        "feature_days": len(have_f & need),
        "missing_days": sorted(need - have_c),
        "sample_contract_miss_total": sample_miss,
        "sample_days_checked": checked,
        "ready": len(need - have_c) == 0 and len(need - have_f) == 0 and sample_miss == 0,
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fp = pd.read_parquet(args.fingerprints)
    fp["month"] = fp["date_str"].astype(str).str.slice(0, 7)
    if "split" not in fp.columns:
        fp["split"] = np.where(fp["date_str"] >= "2026-07-01", "jul_oos", "jan_jun")

    # join regime
    if Path(args.regime_preds).exists():
        reg = pd.read_parquet(args.regime_preds)
        fp = fp.merge(reg[["date_str", "regime_pred", "p_regime_2"]], on="date_str", how="left")

    cov = dte1_coverage(Path(args.dte1_map), Path(args.dte1_micro))

    # --- Protocol A: fit Jan–Apr, eval May–Jun shadow + Jul OOS ---
    fit = fp[fp["month"].isin(["2026-01", "2026-02", "2026-03", "2026-04"])]
    thr_a = fit_thresholds(fit)
    rows = []
    rows += eval_window(fp[fp["month"].isin(["2026-01", "2026-02", "2026-03"])], thr_a, args.position_frac, "jan_mar_insample_diag")
    rows += eval_window(fp[fp["month"].isin(["2026-04"])], thr_a, args.position_frac, "apr_fit_diag")
    rows += eval_window(fp[fp["month"].isin(["2026-05", "2026-06"])], thr_a, args.position_frac, "may_jun_shadow")
    rows += eval_window(fp[fp["split"] == "jul_oos"], thr_a, args.position_frac, "jul_oos")
    rows += eval_window(fp[fp["month"].isin(["2026-04", "2026-05", "2026-06"])], thr_a, args.position_frac, "apr_jun_shadow")

    # with regime block_pred_1 stacked on onset
    if "regime_pred" in fp.columns:
        for window, months in [
            ("may_jun_shadow_block1", ["2026-05", "2026-06"]),
            ("jul_oos_block1", None),
            ("apr_jun_shadow_block1", ["2026-04", "2026-05", "2026-06"]),
        ]:
            if months is None:
                sub = fp[fp["split"] == "jul_oos"].copy()
            else:
                sub = fp[fp["month"].isin(months)].copy()
            sub = sub[sub["regime_pred"].fillna(-1).astype(int) != 1]
            rows += eval_window(sub, thr_a, args.position_frac, window)

    # --- Protocol B: expanding month thresholds ---
    expand_rows = []
    for month in sorted(fp["month"].unique()):
        train = fp[fp["month"] < month]
        test = fp[fp["month"] == month]
        if len(train) < 20 or test.empty:
            continue
        thr = fit_thresholds(train)
        for r in eval_window(test, thr, args.position_frac, f"expand_{month}"):
            r["protocol"] = "expanding"
            expand_rows.append(r)

    gate_df = pd.DataFrame(rows)
    gate_df["protocol"] = "fit_jan_apr"
    expand_df = pd.DataFrame(expand_rows)
    all_gates = pd.concat([gate_df, expand_df], ignore_index=True)
    all_gates.to_csv(out / "gate_results.csv", index=False)

    # 1DTE route candidates: 0DTE days that are weak under baseline but onset says avoid / F2 low
    jj = fp[fp["split"] == "jan_jun"].copy()
    day = (
        jj.groupby("date_str", as_index=False)
        .agg(
            n=("path_exec_ret", "size"),
            avg_ret=("path_exec_ret", "mean"),
            mean_persist=("f3_flow_persist", "mean"),
            mean_accel=("f3_flow_accel", "mean"),
            mean_f2=("f2_rv_vs_prem", "mean"),
            mean_session=("f3_session_minute", "mean"),
        )
    )
    thr = thr_a
    day["high_persist_day"] = day["mean_persist"] >= thr["persist_q67"]
    day["low_f2_day"] = day["mean_f2"] <= day["mean_f2"].quantile(0.33)
    day["late_day"] = day["mean_session"] >= 180
    day["route_1dte_candidate"] = day["low_f2_day"] | (day["avg_ret"] < 0) & (day["high_persist_day"] | day["late_day"])
    # fix operator precedence explicitly
    day["route_1dte_candidate"] = day["low_f2_day"] | ((day["avg_ret"] < 0) & (day["high_persist_day"] | day["late_day"]))
    day["dte1_micro_ready"] = day["date_str"].isin(
        {p.stem.replace("QQQ_", "") for p in (Path(args.dte1_micro) / "contract_1s" / "QQQ").glob("*.parquet")}
    )
    day.to_csv(out / "daily_0dte_route_flags.csv", index=False)

    # headline comparison table
    focus = all_gates[
        all_gates["window"].isin(
            ["may_jun_shadow", "jul_oos", "apr_jun_shadow", "jan_mar_insample_diag", "may_jun_shadow_block1", "jul_oos_block1"]
        )
        & all_gates["gate"].isin(["baseline", "block_high_persist", "ignition_accel", "ignition_only", "onset_prefer", "block_chase_late"])
    ].copy()
    focus.to_csv(out / "headline_gates.csv", index=False)

    # promotion heuristic (no Jul tuning): may_jun shadow improves vs baseline on account or DD, keep >=50% trades;
    # apr_jun not destroyed (<50% account wipe)
    may = focus[focus["window"] == "may_jun_shadow"].set_index("gate")
    apr = focus[focus["window"] == "apr_jun_shadow"].set_index("gate")
    jul = focus[focus["window"] == "jul_oos"].set_index("gate")
    base_may = may.loc["baseline"] if "baseline" in may.index else None
    candidates = []
    if base_may is not None:
        for g in may.index:
            if g == "baseline":
                continue
            row = may.loc[g]
            apr_row = apr.loc[g] if g in apr.index else None
            jul_row = jul.loc[g] if g in jul.index else None
            ok = (
                row["n_keep_ratio"] >= 0.5
                and (row["account_ret"] > base_may["account_ret"] or row["max_dd"] > base_may["max_dd"] - 1e-9)
                and (apr_row is None or apr_row["n_keep_ratio"] >= 0.5)
                and (apr_row is None or apr_row["account_ret"] >= 0.5 * float(apr.loc["baseline", "account_ret"]))
            )
            candidates.append(
                {
                    "gate": g,
                    "promote_shadow": bool(ok),
                    "may_jun": row.to_dict(),
                    "apr_jun": apr_row.to_dict() if apr_row is not None else None,
                    "jul_oos_report_only": jul_row.to_dict() if jul_row is not None else None,
                }
            )

    summary = {
        "experiment": "onset_full_validation",
        "alpha_thesis": "predict institutional onset; never chase high persist",
        "fit_thresholds_jan_apr": thr_a,
        "dte1_coverage": cov,
        "route_1dte_candidate_days": int(day["route_1dte_candidate"].sum()),
        "route_1dte_ready_overlap": int((day["route_1dte_candidate"] & day["dte1_micro_ready"]).sum()),
        "promotion_candidates": [c for c in candidates if c["promote_shadow"]],
        "all_gate_candidates": candidates,
        "headline": focus.to_dict(orient="records"),
        "verdict_notes": [
            "Jul is report-only; do not promote from Jul",
            "ignition_only may be low-n; prefer block_high_persist if it passes shadow",
            "1DTE route candidates need independent 1DTE rules before claiming combo alpha",
        ],
        "files": {
            "gates": str(out / "gate_results.csv"),
            "headline": str(out / "headline_gates.csv"),
            "daily_route": str(out / "daily_0dte_route_flags.csv"),
            "summary": str(out / "summary.json"),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(
        json.dumps(
            {
                "dte1_ready": cov["ready"],
                "dte1_days": cov["contract_days"],
                "fit_thr": thr_a,
                "promotion_candidates": summary["promotion_candidates"],
                "may_jun": may.reset_index().to_dict(orient="records") if not may.empty else [],
                "jul_oos": jul.reset_index().to_dict(orient="records") if not jul.empty else [],
                "route_1dte_days": summary["route_1dte_candidate_days"],
            },
            indent=2,
            default=str,
        )
    )
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
