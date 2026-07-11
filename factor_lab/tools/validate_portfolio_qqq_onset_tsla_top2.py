#!/usr/bin/env python3
"""Validate QQQ onset gate (block_chase_late) + TSLA Top2 parallel portfolio.

QQQ uses path_exec_ret (curated confirm+statehold).
TSLA uses target_exec_ret_30s from independent dte0 rule
  tree_edge × is_underlying_breaking_down × CALL, daily Top2
  (label replay — not path; stated explicitly in summary).

Portfolio rules:
  - Scan both; hold both allowed
  - Per-name weight 0.25; same-day total cap 0.40 (pro-rata if both trade)
  - Optional: also apply block_pred_1 on QQQ days
  - Causal onset thr: fit persist q67 on QQQ Jan–Apr fingerprints

Windows:
  - overlap Feb–Jun (TSLA micro starts ~Feb)
  - Jul: QQQ-only report if fingerprints exist
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from factor_lab.tools.analyze_0dte_rule_state_stability import apply_rule_scorers, fit_rule_scorers
from factor_lab.tools.analyze_0dte_state_alpha_attribution import choose_daily_topk


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--qqq-trades",
        default="factor_lab/results/0dte_state_gate_curated_confirm_statehold_jan_jun_pos25/trades_all.parquet",
    )
    p.add_argument(
        "--qqq-jul",
        default="factor_lab/results/0dte_state_gate_curated_confirm_statehold_jul2026_w1_pos25/trades_all.parquet",
    )
    p.add_argument(
        "--fingerprints",
        default="factor_lab/results/0dte_institutional_fingerprints/trades_with_fingerprints.parquet",
    )
    p.add_argument(
        "--regime-preds",
        default="factor_lab/results/0dte_qqq_deep_anchor_scaffold/causal_regime_predictions.parquet",
    )
    p.add_argument("--name-weight", type=float, default=0.25)
    p.add_argument("--day-cap", type=float, default=0.40)
    p.add_argument(
        "--output-dir",
        default="qqq_btc/results/portfolio_qqq_onset_tsla_top2_validation",
    )
    return p.parse_args()


def account_from_day_returns(day_rets: pd.Series) -> dict:
    r = pd.to_numeric(day_rets, errors="coerce").dropna()
    if r.empty:
        return {"n_days": 0, "avg_day": float("nan"), "account_ret": 0.0, "max_dd": 0.0}
    eq = np.cumprod(1.0 + r.to_numpy())
    peaks = np.maximum.accumulate(np.r_[1.0, eq])[:-1]
    return {
        "n_days": int(len(r)),
        "avg_day": float(r.mean()),
        "account_ret": float(eq[-1] - 1.0),
        "max_dd": float((eq / peaks - 1.0).min()),
    }


def trade_account(returns: pd.Series, pf: float) -> dict:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return {"n": 0, "avg_ret": float("nan"), "account_ret": 0.0, "max_dd": 0.0, "win_rate": float("nan")}
    eq = np.cumprod(1.0 + pf * r.to_numpy())
    peaks = np.maximum.accumulate(np.r_[1.0, eq])[:-1]
    return {
        "n": int(len(r)),
        "avg_ret": float(r.mean()),
        "account_ret": float(eq[-1] - 1.0),
        "max_dd": float((eq / peaks - 1.0).min()),
        "win_rate": float((r > 0).mean()),
    }


def build_tsla_top2() -> pd.DataFrame:
    target = "target_exec_ret_30s"
    keep = [
        "timestamp",
        "date_str",
        "side",
        "side_code",
        "ticker",
        "bucket_id",
        target,
        "time_score",
        "trend_score",
        "flow_score",
        "liquidity_score",
        "vol_score",
        "gamma_score",
        "stock_ret_30s",
        "stock_ret_60s",
        "stock_vwap_dev",
        "spread_pct",
        "score_hot_quote_tight",
        "is_vol_expansion",
        "is_range_pin_proxy",
        "is_put_trend_proxy",
        "is_call_trend_proxy",
        "is_stock_trend_up",
        "is_stock_trend_down",
        "is_stock_vwap_extension",
    ]
    months = ["2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
    fit_months = ["2026-02", "2026-03"]

    def cache(m: str) -> Path:
        return Path(f"stock_options/results/mag7_state_gate_tsla_dte0/cache/score_dataset_{m}.parquet")

    sample_cols = set(pd.read_parquet(cache("2026-02")).columns)
    use = [c for c in keep if c in sample_cols]
    fit = pd.concat(
        [pd.read_parquet(cache(m), columns=use).assign(month=m) for m in fit_months if cache(m).exists()],
        ignore_index=True,
    )
    fit_s = fit.sample(n=min(250000, len(fit)), random_state=42)
    _, weights, model = fit_rule_scorers(fit_s, target)

    frames = []
    for m in months:
        fp = cache(m)
        if not fp.exists():
            continue
        df = pd.read_parquet(fp, columns=use)
        df = apply_rule_scorers(df, weights, model)
        bd = (
            (pd.to_numeric(df["stock_ret_30s"], errors="coerce").fillna(0) < 0)
            & (pd.to_numeric(df["stock_ret_60s"], errors="coerce").fillna(0) < 0)
            & (pd.to_numeric(df["stock_vwap_dev"], errors="coerce").fillna(0) < 0)
        )
        sub = df[bd & df["side"].astype(str).str.upper().eq("CALL")].copy()
        if sub.empty:
            continue
        picks = choose_daily_topk(sub, "tree_edge_score", max_topk=2, cooldown_s=30)
        picks = picks[pd.to_numeric(picks["pick_rank"], errors="coerce") <= 2].copy()
        picks["month"] = m
        picks["symbol"] = "TSLA"
        picks["ret"] = pd.to_numeric(picks[target], errors="coerce")
        frames.append(picks)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.tz_convert("America/New_York")
        open_ts = out["timestamp"].dt.normalize() + pd.Timedelta(hours=9, minutes=30)
        # fix: use date-based open
        out["session_minute"] = [
            (ts - pd.Timestamp(f"{ts.date()} 09:30:00", tz=ts.tz)).total_seconds() / 60.0 for ts in out["timestamp"]
        ]
    return out


def gate_qqq(fp: pd.DataFrame, thr_persist: float, late: float = 180.0) -> pd.DataFrame:
    persist = pd.to_numeric(fp["f3_flow_persist"], errors="coerce")
    sess = pd.to_numeric(fp["f3_session_minute"], errors="coerce")
    keep = ~((persist >= thr_persist) & (sess >= late))
    return fp.loc[keep].copy()


def portfolio_day_returns(
    qqq: pd.DataFrame,
    tsla: pd.DataFrame,
    *,
    name_w: float,
    day_cap: float,
    qqq_ret_col: str = "path_exec_ret",
    tsla_ret_col: str = "ret",
) -> pd.DataFrame:
    """Build ordered daily portfolio returns with same-day cap."""
    rows = []
    qqq = qqq.copy()
    tsla = tsla.copy()
    qqq["symbol"] = "QQQ"
    qqq["_ret"] = pd.to_numeric(qqq[qqq_ret_col], errors="coerce")
    tsla["_ret"] = pd.to_numeric(tsla[tsla_ret_col], errors="coerce")
    days = sorted(set(qqq["date_str"]).union(set(tsla["date_str"])))
    for d in days:
        q = qqq[qqq["date_str"] == d]
        t = tsla[tsla["date_str"] == d]
        legs = []
        if not q.empty:
            legs.append(("QQQ", float(q["_ret"].mean()), len(q)))  # day avg within name
        if not t.empty:
            legs.append(("TSLA", float(t["_ret"].mean()), len(t)))
        if not legs:
            continue
        # allocate weights
        if len(legs) == 1:
            w = {legs[0][0]: min(name_w, day_cap)}
        else:
            raw = {s: name_w for s, _, _ in legs}
            ssum = sum(raw.values())
            if ssum > day_cap:
                scale = day_cap / ssum
                raw = {k: v * scale for k, v in raw.items()}
            w = raw
        day_ret = sum(w[s] * r for s, r, _ in legs)
        rows.append(
            {
                "date_str": d,
                "day_ret": day_ret,
                "n_qqq": int(len(q)),
                "n_tsla": int(len(t)),
                "w_qqq": float(w.get("QQQ", 0.0)),
                "w_tsla": float(w.get("TSLA", 0.0)),
                "qqq_avg": float(q["_ret"].mean()) if len(q) else float("nan"),
                "tsla_avg": float(t["_ret"].mean()) if len(t) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("date_str")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[port] build TSLA top2", flush=True)
    tsla = build_tsla_top2()
    tsla.to_parquet(out / "tsla_top2_label30s.parquet", index=False)
    print(f"[port] TSLA top2 n={len(tsla)} days={tsla['date_str'].nunique() if len(tsla) else 0}", flush=True)

    fp = pd.read_parquet(args.fingerprints)
    fp["month"] = fp["date_str"].astype(str).str.slice(0, 7)
    fit = fp[fp["month"].isin(["2026-01", "2026-02", "2026-03", "2026-04"])]
    thr_p = float(pd.to_numeric(fit["f3_flow_persist"], errors="coerce").quantile(0.67))
    print(f"[port] onset thr persist_q67={thr_p:.4f}", flush=True)

    qqq_all = fp[fp["split"] == "jan_jun"].copy()
    qqq_onset = gate_qqq(qqq_all, thr_p)
    # regime optional
    if Path(args.regime_preds).exists():
        reg = pd.read_parquet(args.regime_preds)
        qqq_onset_r = qqq_onset.merge(reg[["date_str", "regime_pred"]], on="date_str", how="left")
        qqq_onset_r = qqq_onset_r[qqq_onset_r["regime_pred"].fillna(-1).astype(int) != 1]
    else:
        qqq_onset_r = qqq_onset

    # overlap window Feb-Jun
    def in_overlap(df: pd.DataFrame) -> pd.DataFrame:
        return df[(df["date_str"] >= "2026-02-02") & (df["date_str"] <= "2026-06-30")].copy()

    tsla_o = in_overlap(tsla)
    configs = {
        "qqq_baseline": in_overlap(qqq_all),
        "qqq_onset": in_overlap(qqq_onset),
        "qqq_onset_block1": in_overlap(qqq_onset_r),
        "tsla_top2": tsla_o,
    }

    single = {}
    for name, df in configs.items():
        ret_col = "ret" if name.startswith("tsla") else "path_exec_ret"
        single[name] = trade_account(df[ret_col], args.name_weight)

    # portfolios
    ports = {}
    day_tables = {}
    for pname, qdf in [
        ("qqq_base_+_tsla", in_overlap(qqq_all)),
        ("qqq_onset_+_tsla", in_overlap(qqq_onset)),
        ("qqq_onset_block1_+_tsla", in_overlap(qqq_onset_r)),
    ]:
        days = portfolio_day_returns(qdf, tsla_o, name_w=args.name_weight, day_cap=args.day_cap)
        day_tables[pname] = days
        days.to_csv(out / f"daily_{pname}.csv", index=False)
        ports[pname] = account_from_day_returns(days["day_ret"])
        ports[pname]["n_qqq_trades"] = int(len(qdf))
        ports[pname]["n_tsla_trades"] = int(len(tsla_o))
        ports[pname]["overlap_days_both"] = int(((days["n_qqq"] > 0) & (days["n_tsla"] > 0)).sum())

    # Jul QQQ-only report
    jul = fp[fp["split"] == "jul_oos"].copy()
    jul_onset = gate_qqq(jul, thr_p) if not jul.empty else jul
    jul_stats = {
        "qqq_jul_baseline": trade_account(jul["path_exec_ret"], args.name_weight) if len(jul) else {},
        "qqq_jul_onset": trade_account(jul_onset["path_exec_ret"], args.name_weight) if len(jul_onset) else {},
    }

    # monthly portfolio attribution for best candidate
    best_name = "qqq_onset_+_tsla"
    days = day_tables[best_name]
    days["month"] = days["date_str"].str.slice(0, 7)
    monthly = []
    for m, g in days.groupby("month"):
        st = account_from_day_returns(g["day_ret"])
        st["month"] = m
        monthly.append(st)

    # promotion: vs qqq baseline portfolio alone on same days (day-compound with name_w)
    qqq_base_days = portfolio_day_returns(
        in_overlap(qqq_all), tsla_o.iloc[0:0], name_w=args.name_weight, day_cap=args.day_cap
    )
    qqq_onset_days = portfolio_day_returns(
        in_overlap(qqq_onset), tsla_o.iloc[0:0], name_w=args.name_weight, day_cap=args.day_cap
    )
    ports["qqq_baseline_day"] = account_from_day_returns(qqq_base_days["day_ret"])
    ports["qqq_onset_day"] = account_from_day_returns(qqq_onset_days["day_ret"])
    ports["tsla_only_day"] = account_from_day_returns(
        portfolio_day_returns(qqq_all.iloc[0:0], tsla_o, name_w=args.name_weight, day_cap=args.day_cap)["day_ret"]
    )

    promote = (
        ports["qqq_onset_+_tsla"]["account_ret"] > ports["qqq_baseline_day"]["account_ret"]
        and ports["qqq_onset_+_tsla"]["max_dd"] >= ports["qqq_baseline_day"]["max_dd"] - 0.05
    )

    summary = {
        "experiment": "portfolio_qqq_onset_tsla_top2",
        "window": "2026-02-02..2026-06-30",
        "weights": {"name_weight": args.name_weight, "day_cap": args.day_cap},
        "onset_thr_persist_q67": thr_p,
        "tsla_note": "label target_exec_ret_30s; rule tree_edge×breaking_down CALL top2; NOT path replay",
        "single_name_trade_compound": single,
        "portfolio_day_compound": ports,
        "monthly_qqq_onset_tsla": monthly,
        "jul_qqq_only_report": jul_stats,
        "promote_shadow": bool(promote),
        "verdict": (
            "shadow_promote_qqq_onset_plus_tsla_top2"
            if promote
            else "no_promote_keep_components_separate"
        ),
        "files": {
            "tsla": str(out / "tsla_top2_label30s.parquet"),
            "summary": str(out / "summary.json"),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(
        json.dumps(
            {
                "promote_shadow": summary["promote_shadow"],
                "verdict": summary["verdict"],
                "portfolio": {k: ports[k] for k in ports},
                "single": single,
                "jul": jul_stats,
                "monthly": monthly,
            },
            indent=2,
            default=str,
        )
    )
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
