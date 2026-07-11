#!/usr/bin/env python3
"""Curated State Gate replay for QQQ 0DTE.

Only activate historically stable alpha rules inside their states:
  - tree_edge_score | is_qqq_recovering | ALL | top1
  - tree_edge_score | is_stock_trend_down__and__is_lunch | ALL | top1

Default confirmation filters (Apr+May survivors, Jun OOS positive):
  - recovering: is_put_trend_proxy
  - lunch: flow_score >= train median (fit on confirm-fit months)

Default exit uses state clock (not hand-weighted RightTailScore):
  - recovering: 45s
  - lunch: 180s

Entry ranking uses tree_edge_score. Exit uses path-level ask->bid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from factor_lab.tools.analyze_0dte_rule_state_stability import (
    apply_rule_scorers,
    attach_all_states,
    fit_rule_scorers,
    load_fit_period,
    load_or_build_month,
)
from factor_lab.tools.analyze_0dte_state_alpha_attribution import choose_daily_topk, replay_metrics
from factor_lab.tools.run_0dte_adaptive_rule_pool import (
    adaptive_daily_trades,
    fixed_baseline,
    idle_ratio,
    rule_active_mask,
    summarize_trades,
)


CURATED_RULES = pd.DataFrame(
    [
        {
            "rule": "tree_edge_score",
            "state": "is_qqq_recovering",
            "side": "ALL",
            "topk_per_day": 1,
            "name": "recovering_reversal",
        },
        {
            "rule": "tree_edge_score",
            "state": "is_stock_trend_down__and__is_lunch",
            "side": "ALL",
            "topk_per_day": 1,
            "name": "trend_down_lunch",
        },
    ]
)

# Frozen confirmation policy. Thresholds for continuous filters are fit on
# --confirm-fit-months only (default Apr+May), never on Jun.
CONFIRM_SPECS = {
    "is_qqq_recovering": {"kind": "flag", "col": "is_put_trend_proxy"},
    "is_stock_trend_down__and__is_lunch": {
        "kind": "score_ge_q",
        "col": "flow_score",
        "q": 0.50,
    },
}

# State-specific hold clock (from RightTail / MFE study):
# recovering peaks earlier -> 45s; lunch trend continuation -> 180s.
# Hand-weighted RightTailScore v0 failed; supervised v1 is optional overlay.
STATE_HOLD_S = {
    "is_qqq_recovering": 45,
    "is_stock_trend_down__and__is_lunch": 180,
}

DEFAULT_RIGHT_TAIL_V1_MODEL = (
    "factor_lab/results/0dte_state_gate_right_tail_v1_apr_jun/right_tail_v1_logit_extend_helps.joblib"
)


def month_bounds(month: str) -> tuple[str, str]:
    start = f"{month}-01"
    end = (pd.Timestamp(start) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
    if month == "2026-04":
        start = "2026-04-13"
    return start, end


def load_month_panel(args: argparse.Namespace, month: str, target: str, thresholds, weights, model, cache_dir: Path) -> pd.DataFrame:
    start, end = month_bounds(month)
    data = load_or_build_month(args, month, start, end, target, thresholds, cache_dir)
    data = apply_rule_scorers(data, weights, model)
    data, _ = attach_all_states(data)
    return data


def fit_confirm_thresholds(trades: pd.DataFrame, specs: dict[str, dict] | None = None) -> dict[str, float]:
    """Fit continuous confirmation thresholds on confirm-fit curated trades only.

    Quantiles are fit on all confirm-fit trades (same as the confirm study),
    then applied per-state.
    """
    specs = specs or CONFIRM_SPECS
    if trades is None or trades.empty:
        return {}
    thr: dict[str, float] = {}
    for state, spec in specs.items():
        if spec.get("kind") != "score_ge_q":
            continue
        col = spec["col"]
        q = float(spec.get("q", 0.50))
        if col not in trades.columns:
            continue
        s = pd.to_numeric(trades[col], errors="coerce").dropna()
        if s.empty:
            continue
        thr[f"{state}::{col}"] = float(s.quantile(q))
    return thr


def confirm_mask(
    df: pd.DataFrame,
    state: str,
    *,
    thresholds: dict[str, float],
    specs: dict[str, dict] | None = None,
) -> pd.Series:
    """Return confirmation mask for one curated state. Missing spec => pass-through."""
    specs = specs or CONFIRM_SPECS
    if state not in specs:
        return pd.Series(True, index=df.index)
    spec = specs[state]
    kind = spec["kind"]
    col = spec["col"]
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = pd.to_numeric(df[col], errors="coerce")
    if kind == "flag":
        return s.fillna(0.0) > 0.5
    if kind == "score_ge_q":
        key = f"{state}::{col}"
        if key not in thresholds:
            return pd.Series(False, index=df.index)
        return s >= float(thresholds[key])
    raise ValueError(f"unknown confirm kind: {kind}")


def filter_trades_by_confirm(
    trades: pd.DataFrame,
    *,
    thresholds: dict[str, float],
    enabled: bool,
) -> pd.DataFrame:
    """Drop selected trades that fail state confirmation. No TopK refill.

    Matches the walk-forward confirm study: filter after selection so weak
    alternatives do not replace rejected high-edge picks.
    """
    if not enabled or trades.empty or "active_state" not in trades.columns:
        return trades
    keep = pd.Series(False, index=trades.index)
    for state, g in trades.groupby("active_state"):
        keep.loc[g.index] = confirm_mask(g, state, thresholds=thresholds).to_numpy()
    out = trades.loc[keep].copy()
    if not out.empty:
        out["confirm_passed"] = True
    return out


def load_right_tail_v1(model_path: str | Path) -> dict:
    import joblib

    blob = joblib.load(model_path)
    if isinstance(blob, dict) and "model" in blob:
        return blob
    return {"model": blob, "meta": {}}


def score_right_tail_v1(trades: pd.DataFrame, artifact: dict) -> pd.Series:
    """Score confirmed trades with RightTail v1 logit (extend_helps_180).

    Only entry-time features are required; path MFE columns are optional.
    """
    from factor_lab.tools.analyze_0dte_state_gate_right_tail_v1 import feature_matrix

    if trades.empty:
        return pd.Series(dtype=float)
    prepared = trades.copy()
    prepared["is_recovering_state"] = prepared["active_state"].eq("is_qqq_recovering").astype(float)
    prepared["is_lunch_state"] = prepared["active_state"].eq("is_stock_trend_down__and__is_lunch").astype(float)
    prepared["is_call_side"] = prepared["side"].eq("CALL").astype(float)
    x = feature_matrix(prepared)
    cols = list(artifact.get("meta", {}).get("feature_cols") or x.columns)
    x = x.reindex(columns=cols, fill_value=0.0)
    model = artifact["model"]
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x)[:, 1]
    else:
        scores = model.predict(x)
    return pd.Series(scores, index=trades.index, name="right_tail_v1_score")


def attach_right_tail_hold(
    trades: pd.DataFrame,
    *,
    artifact: dict,
    short_hold_s: int = 45,
    long_hold_s: int = 180,
) -> pd.DataFrame:
    """If score >= train threshold, hold long_hold_s else short_hold_s."""
    if trades.empty:
        return trades
    out = trades.copy()
    scores = score_right_tail_v1(out, artifact)
    thr = float(artifact.get("meta", {}).get("threshold", 0.5))
    high = scores >= thr
    out["right_tail_v1_score"] = scores
    out["right_tail_v1_high"] = high.astype(float)
    out["hold_s_override"] = np.where(high, int(long_hold_s), int(short_hold_s)).astype(int)
    return out


def resolve_hold_s(
    trade,
    *,
    default_hold_s: int,
    state_hold: dict[str, int] | None,
    use_state_hold: bool,
) -> int:
    # Explicit per-trade override from RightTail v1 takes priority.
    override = getattr(trade, "hold_s_override", None)
    if override is not None and np.isfinite(override):
        return int(override)
    if use_state_hold and state_hold:
        state = str(getattr(trade, "active_state", "") or "")
        if state in state_hold:
            return int(state_hold[state])
    return int(default_hold_s)


def path_exec_return(
    panel: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    hold_s: int,
    commission: float,
    state_hold: dict[str, int] | None = None,
    use_state_hold: bool = False,
) -> pd.DataFrame:
    """Replace label target with path-level ask->bid return after hold_s seconds.

    When use_state_hold=True, each trade uses STATE_HOLD_S[active_state] if present.
    """
    if trades.empty:
        return trades
    out_rows = []
    by_ticker = {t: g.sort_values("timestamp").reset_index(drop=True) for t, g in panel.groupby("ticker", sort=False)}
    for tr in trades.itertuples(index=False):
        rec = tr._asdict()
        path = by_ticker.get(str(getattr(tr, "ticker")))
        if path is None or path.empty:
            continue
        ts = pd.Timestamp(getattr(tr, "timestamp"))
        ts_ns = pd.to_datetime(path["timestamp"]).astype("int64").to_numpy()
        pos = int(np.searchsorted(ts_ns, ts.value, side="left"))
        if pos >= len(path):
            continue
        if abs(int(ts_ns[pos]) - ts.value) > 1_500_000_000:
            exact = np.where(ts_ns == ts.value)[0]
            if len(exact) == 0:
                continue
            pos = int(exact[0])
        trade_hold = resolve_hold_s(
            tr,
            default_hold_s=hold_s,
            state_hold=state_hold,
            use_state_hold=use_state_hold,
        )
        exit_pos = pos + int(trade_hold)
        if exit_pos >= len(path):
            continue
        entry_ask = float(pd.to_numeric(path["ask"].iloc[pos], errors="coerce"))
        exit_bid = float(pd.to_numeric(path["bid"].iloc[exit_pos], errors="coerce"))
        if not np.isfinite(entry_ask) or not np.isfinite(exit_bid) or entry_ask <= 0:
            continue
        cost = 2.0 * commission / (entry_ask * 100.0)
        ret = exit_bid / entry_ask - 1.0 - cost
        rec["hold_s"] = int(trade_hold)
        rec["entry_ask"] = entry_ask
        rec["exit_bid"] = exit_bid
        rec["path_exec_ret"] = float(ret)
        out_rows.append(rec)
    return pd.DataFrame(out_rows)


def single_rule_replay(
    df: pd.DataFrame,
    rule_row: pd.Series,
    *,
    hold_s: int,
    cooldown_s: int,
    commission: float,
) -> dict:
    sub = df.loc[rule_active_mask(df, rule_row["state"], rule_row["side"])]
    picks = choose_daily_topk(sub, rule_row["rule"], max_topk=int(rule_row["topk_per_day"]), cooldown_s=cooldown_s)
    if picks.empty:
        metrics = {"trades": 0}
    else:
        path_picks = path_exec_return(df, picks, hold_s=hold_s, commission=commission)
        if path_picks.empty:
            metrics = {"trades": 0}
        else:
            path_picks = path_picks.copy()
            path_picks["pick_rank"] = 1
            metrics = replay_metrics(path_picks, "path_exec_ret", int(rule_row["topk_per_day"]))
    return {
        "name": rule_row.get("name", rule_row["state"]),
        "rule": rule_row["rule"],
        "state": rule_row["state"],
        "side": rule_row["side"],
        "topk_per_day": int(rule_row["topk_per_day"]),
        "hold_s": int(hold_s),
        "active_rows": int(len(sub)),
        "active_row_ratio": float(len(sub) / max(len(df), 1)),
        **metrics,
    }


def evaluate_month(
    df: pd.DataFrame,
    *,
    hold_s: int,
    cooldown_s: int,
    daily_topk: int,
    commission: float,
    confirm_thresholds: dict[str, float] | None = None,
    enable_confirm: bool = True,
    state_hold: dict[str, int] | None = None,
    use_state_hold: bool = True,
    right_tail_artifact: dict | None = None,
) -> dict:
    # Ranking uses tree_edge_score; PnL uses path hold execution.
    # Confirmation is applied AFTER TopK selection (no refill).
    # Default exit: state clock (recovering=45s, lunch=180s).
    # Optional: --right-tail-v1 overrides hold via logit extend_helps score.
    thr = confirm_thresholds or {}
    sh = state_hold or STATE_HOLD_S
    gate_trades = adaptive_daily_trades(df, CURATED_RULES, "tree_edge_score", cooldown_s, daily_topk)
    gate_trades = filter_trades_by_confirm(gate_trades, thresholds=thr, enabled=enable_confirm)
    if right_tail_artifact is not None and not gate_trades.empty:
        gate_trades = attach_right_tail_hold(gate_trades, artifact=right_tail_artifact)
    gate_trades = path_exec_return(
        df,
        gate_trades,
        hold_s=hold_s,
        commission=commission,
        state_hold=sh,
        use_state_hold=use_state_hold and right_tail_artifact is None,
    )
    baseline = fixed_baseline(df, "tree_edge_score", cooldown_s)
    baseline = path_exec_return(df, baseline, hold_s=hold_s, commission=commission)
    per_rule = []
    for _, row in CURATED_RULES.iterrows():
        sub = df.loc[rule_active_mask(df, row["state"], row["side"])]
        picks = choose_daily_topk(sub, row["rule"], max_topk=int(row["topk_per_day"]), cooldown_s=cooldown_s)
        if not picks.empty:
            picks = picks.copy()
            picks["active_state"] = row["state"]
            picks = filter_trades_by_confirm(picks, thresholds=thr, enabled=enable_confirm)
            if right_tail_artifact is not None and not picks.empty:
                picks = attach_right_tail_hold(picks, artifact=right_tail_artifact)
        if picks.empty:
            metrics = {"trades": 0}
            used_hold = int(sh.get(row["state"], hold_s)) if use_state_hold else int(hold_s)
        else:
            path_picks = path_exec_return(
                df,
                picks,
                hold_s=hold_s,
                commission=commission,
                state_hold=sh,
                use_state_hold=use_state_hold and right_tail_artifact is None,
            )
            if path_picks.empty:
                metrics = {"trades": 0}
                used_hold = int(sh.get(row["state"], hold_s)) if use_state_hold else int(hold_s)
            else:
                path_picks = path_picks.copy()
                path_picks["pick_rank"] = 1
                metrics = replay_metrics(path_picks, "path_exec_ret", int(row["topk_per_day"]))
                used_hold = int(path_picks["hold_s"].iloc[0]) if "hold_s" in path_picks.columns else int(hold_s)
        per_rule.append(
            {
                "name": row.get("name", row["state"]),
                "rule": row["rule"],
                "state": row["state"],
                "side": row["side"],
                "topk_per_day": int(row["topk_per_day"]),
                "hold_s": used_hold,
                "active_rows": int(len(sub)),
                "active_row_ratio": float(len(sub) / max(len(df), 1)),
                **metrics,
            }
        )
    return {
        "rows": int(len(df)),
        "days": int(df["date_str"].nunique()),
        "state_gate": {
            **summarize_trades(gate_trades, "path_exec_ret", "state_gate"),
            "idle_day_ratio": idle_ratio(df, gate_trades),
            "active_day_ratio": 1.0 - idle_ratio(df, gate_trades),
        },
        "fixed_call_trend_down": summarize_trades(baseline, "path_exec_ret", "fixed_call_trend_down"),
        "per_rule": per_rule,
        "trades": gate_trades,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--micro-root", default="/mnt/s990/data/microstructure/qqq_0dte_api_ladder")
    p.add_argument("--stock-root", default="/mnt/s990/data/raw_1s/stocks/QQQ")
    p.add_argument("--fit-start", default="2026-04-13")
    p.add_argument("--fit-end", default="2026-04-30")
    p.add_argument("--cache-dir", default="factor_lab/results/0dte_rule_state_stability_apr_jun/cache")
    p.add_argument("--months", default="2026-04,2026-05,2026-06")
    p.add_argument("--confirm-fit-months", default="2026-04,2026-05", help="months used to fit confirm thresholds")
    p.add_argument(
        "--confirm-thresholds-json",
        default="",
        help="optional frozen confirm thresholds JSON; skips refitting when set",
    )
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--lookback-s", type=int, default=60)
    p.add_argument("--horizon-s", type=int, default=30, help="legacy label horizon used while fitting scorers")
    p.add_argument("--hold-s", type=int, default=45, help="fallback hold when state clock disabled")
    p.add_argument("--commission-per-contract", type=float, default=0.65)
    p.add_argument("--max-spread-pct", type=float, default=0.05)
    p.add_argument("--min-ask", type=float, default=0.20)
    p.add_argument("--cooldown-s", type=int, default=30)
    p.add_argument("--daily-topk", type=int, default=2)
    p.add_argument("--no-confirm", action="store_true", help="disable confirmation filters")
    p.add_argument("--no-state-hold", action="store_true", help="use fixed --hold-s for all states")
    p.add_argument(
        "--right-tail-v1",
        action="store_true",
        help="optional overlay: logit extend_helps score gates 45s vs 180s (experimental)",
    )
    p.add_argument("--right-tail-v1-model", default=DEFAULT_RIGHT_TAIL_V1_MODEL)
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--output-dir", default="factor_lab/results/0dte_state_gate_curated_confirm")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    target = f"target_exec_ret_{args.horizon_s}s"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    months = [m.strip() for m in args.months.split(",") if m.strip()]
    confirm_fit_months = [m.strip() for m in args.confirm_fit_months.split(",") if m.strip()]
    enable_confirm = not args.no_confirm
    use_state_hold = not args.no_state_hold
    right_tail_artifact = None
    if args.right_tail_v1:
        right_tail_artifact = load_right_tail_v1(args.right_tail_v1_model)
        print(
            f"[state-gate] right_tail_v1 enabled model={args.right_tail_v1_model} "
            f"thr={right_tail_artifact.get('meta', {}).get('threshold')}",
            flush=True,
        )

    print("[state-gate] fitting scorers", flush=True)
    fit_data, thresholds = load_fit_period(args, target)
    _, weights, model = fit_rule_scorers(fit_data, target)

    # Load all months once; fit confirm thresholds on unconfirmed curated
    # trades from confirm-fit months (same distribution as the confirm study).
    panels: dict[str, pd.DataFrame] = {}
    for month in months:
        print(f"[state-gate] loading {month}", flush=True)
        panels[month] = load_month_panel(args, month, target, thresholds, weights, model, cache_dir)

    confirm_thresholds: dict[str, float] = {}
    if enable_confirm and args.confirm_thresholds_json:
        confirm_thresholds = {
            str(k): float(v) for k, v in json.loads(args.confirm_thresholds_json).items()
        }
        print(f"[state-gate] using frozen confirm thresholds={confirm_thresholds}", flush=True)
    elif enable_confirm:
        fit_trade_frames = []
        for month in confirm_fit_months:
            if month not in panels:
                continue
            raw = adaptive_daily_trades(
                panels[month],
                CURATED_RULES,
                "tree_edge_score",
                args.cooldown_s,
                args.daily_topk,
            )
            if not raw.empty:
                fit_trade_frames.append(raw)
        fit_trades = pd.concat(fit_trade_frames, ignore_index=True) if fit_trade_frames else pd.DataFrame()
        confirm_thresholds = fit_confirm_thresholds(fit_trades)
        print(
            f"[state-gate] confirm fit trades={len(fit_trades)} thresholds={confirm_thresholds}",
            flush=True,
        )
    print(
        f"[state-gate] state_hold={use_state_hold} map={STATE_HOLD_S if use_state_hold else 'disabled'}",
        flush=True,
    )

    monthly = {}
    trade_frames = []
    for month in months:
        panel = panels[month]
        result = evaluate_month(
            panel,
            hold_s=args.hold_s,
            cooldown_s=args.cooldown_s,
            daily_topk=args.daily_topk,
            commission=args.commission_per_contract,
            confirm_thresholds=confirm_thresholds,
            enable_confirm=enable_confirm,
            state_hold=STATE_HOLD_S,
            use_state_hold=use_state_hold,
            right_tail_artifact=right_tail_artifact,
        )
        trades = result.pop("trades")
        if not trades.empty:
            trades = trades.copy()
            trades["month"] = month
            trade_frames.append(trades)
            trades.to_parquet(out_dir / f"trades_{month}.parquet", index=False)
        monthly[month] = result
        print(
            f"[state-gate] {month} confirm={enable_confirm} state_hold={use_state_hold}: "
            f"gate avg={result['state_gate'].get('avg_return', 0):.4f} "
            f"pf={result['state_gate'].get('profit_factor', 0):.3f} "
            f"trades={result['state_gate'].get('trades', 0)} "
            f"vs fixed={result['fixed_call_trend_down'].get('avg_return', 0):.4f}",
            flush=True,
        )

    all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    all_panel_days = sum(v["days"] for v in monthly.values())
    combined = summarize_trades(all_trades, "path_exec_ret", "state_gate_all_months") if not all_trades.empty else {"trades": 0}
    if not all_trades.empty:
        combined["idle_day_ratio"] = float(1.0 - all_trades["date_str"].nunique() / max(all_panel_days, 1))
        combined["active_day_ratio"] = float(all_trades["date_str"].nunique() / max(all_panel_days, 1))
        if "hold_s" in all_trades.columns:
            combined["hold_counts"] = all_trades["hold_s"].value_counts().sort_index().to_dict()
        all_trades.to_parquet(out_dir / "trades_all.parquet", index=False)

    summary = {
        "config": vars(args),
        "curated_rules": CURATED_RULES.to_dict("records"),
        "confirm_specs": CONFIRM_SPECS,
        "confirm_thresholds": confirm_thresholds,
        "confirm_enabled": enable_confirm,
        "confirm_mode": "post_selection_no_refill",
        "state_hold_enabled": use_state_hold and right_tail_artifact is None,
        "state_hold_s": STATE_HOLD_S if (use_state_hold and right_tail_artifact is None) else {},
        "right_tail_v1_enabled": right_tail_artifact is not None,
        "right_tail_v1_meta": (right_tail_artifact or {}).get("meta", {}),
        "selection_note": (
            "These two rules are the only Apr+May positive state rules that remained "
            "positive in Jun on the Rule x State x Month matrix. Gate is frozen; no Jun fitting. "
            "Default confirms (post-selection, no TopK refill): recovering requires "
            "is_put_trend_proxy; lunch requires flow_score >= confirm-fit median. "
            "Default exit uses state clock: recovering=45s, lunch=180s. "
            "Optional --right-tail-v1 overlays logit(extend_helps_180) to choose 45s vs 180s; "
            "experimental until more OOS months."
        ),
        "monthly": monthly,
        "combined": combined,
        "files": {
            "trades_all": str(out_dir / "trades_all.parquet"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(
        json.dumps(
            {
                "confirm_enabled": enable_confirm,
                "state_hold_enabled": use_state_hold and right_tail_artifact is None,
                "state_hold_s": STATE_HOLD_S if (use_state_hold and right_tail_artifact is None) else {},
                "right_tail_v1_enabled": right_tail_artifact is not None,
                "right_tail_v1_meta": (right_tail_artifact or {}).get("meta", {}),
                "confirm_thresholds": confirm_thresholds,
                "monthly": {k: {"gate": v["state_gate"], "fixed": v["fixed_call_trend_down"]} for k, v in monthly.items()},
                "combined": combined,
            },
            indent=2,
            default=str,
        )
    )
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
