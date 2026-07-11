#!/usr/bin/env python3
"""TopK / cross-symbol frequency ablation on MAG7 dte0 caches.

Compares:
  A) per-symbol daily Top1 / Top2 / Top3 inside a frozen rule×state
  B) NVDA+TSLA merged cross-symbol daily Top1 / Top2 / Top3
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
)
from factor_lab.tools.analyze_0dte_state_alpha_attribution import choose_daily_topk, replay_metrics
from factor_lab.tools.run_0dte_adaptive_rule_pool import summarize_trades


# Candidate rules from dte0 stability (independent per symbol).
SYMBOL_RULES = {
    "TSLA": [
        {"rule": "tree_edge_score", "state": "is_underlying_breaking_down", "side": "CALL"},
        {"rule": "tree_edge_score", "state": "is_underlying_recovering", "side": "CALL"},
    ],
    "NVDA": [
        {"rule": "time_score", "state": "is_underlying_recovering", "side": "CALL"},
        {"rule": "tree_edge_score", "state": "is_power_hour", "side": "CALL"},
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="NVDA,TSLA")
    p.add_argument("--selected-dte", type=int, default=0)
    p.add_argument("--fit-months", default="2026-02,2026-03")
    p.add_argument("--eval-months", default="2026-02,2026-03,2026-04,2026-05,2026-06")
    p.add_argument("--cooldown-s", type=int, default=30)
    p.add_argument("--horizon-s", type=int, default=30)
    p.add_argument("--position-frac", type=float, default=0.25)
    p.add_argument(
        "--cache-template",
        default="stock_options/results/mag7_state_gate_{symbol}_dte{dte}/cache",
    )
    p.add_argument(
        "--output-dir",
        default="stock_options/results/mag7_topk_cross_symbol_ablation_dte0",
    )
    return p.parse_args()


def load_symbol_panel(args: argparse.Namespace, symbol: str, months: list[str]) -> pd.DataFrame:
    cache = Path(args.cache_template.format(symbol=symbol.lower(), dte=args.selected_dte))
    frames = []
    for m in months:
        fp = cache / f"score_dataset_{m}.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        df["month"] = m
        df["symbol"] = symbol
        frames.append(df)
    if not frames:
        raise SystemExit(f"no cache for {symbol} under {cache}")
    return pd.concat(frames, ignore_index=True)


def prepare_symbol(args: argparse.Namespace, symbol: str) -> pd.DataFrame:
    fit_months = [x.strip() for x in args.fit_months.split(",") if x.strip()]
    eval_months = [x.strip() for x in args.eval_months.split(",") if x.strip()]
    all_months = sorted(set(fit_months + eval_months))
    panel = load_symbol_panel(args, symbol, all_months)
    target = f"target_exec_ret_{args.horizon_s}s"
    fit = panel[panel["month"].isin(fit_months)].copy()
    _, weights, model = fit_rule_scorers(fit, target)
    panel = apply_rule_scorers(panel, weights, model)
    panel, _ = attach_all_states(panel)
    # ensure time_score exists (used by NVDA candidate)
    if "time_score" not in panel.columns:
        if "tod_frac" in panel.columns:
            panel["time_score"] = pd.to_numeric(panel["tod_frac"], errors="coerce")
        elif "score_hot_quote_tight" in panel.columns:
            panel["time_score"] = pd.to_numeric(panel["score_hot_quote_tight"], errors="coerce")
    return panel


def filter_rule(df: pd.DataFrame, rule: str, state: str, side: str) -> pd.DataFrame:
    out = df.copy()
    if state != "ALL":
        out = out[pd.to_numeric(out.get(state), errors="coerce").fillna(0) > 0.5]
    if side != "ALL":
        out = out[out["side"].astype(str).str.upper().eq(side)]
    if rule not in out.columns:
        return out.iloc[0:0].copy()
    return out


def account_stats(trades: pd.DataFrame, target: str, position_frac: float) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "days": 0,
            "trades_per_active_day": 0.0,
            "avg_return": 0.0,
            "hit_rate": 0.0,
            "total_return_position": 0.0,
        }
    base = summarize_trades(trades, target, "x", position_frac=position_frac)
    days = int(trades["date_str"].nunique())
    return {
        **base,
        "days": days,
        "trades_per_active_day": float(len(trades) / days) if days else 0.0,
    }


def per_symbol_topk(
    panel: pd.DataFrame,
    *,
    rule: str,
    state: str,
    side: str,
    topk: int,
    cooldown_s: int,
    target: str,
    position_frac: float,
) -> tuple[dict, pd.DataFrame]:
    sub = filter_rule(panel, rule, state, side)
    if sub.empty:
        return account_stats(pd.DataFrame(), target, position_frac), pd.DataFrame()
    picks = choose_daily_topk(sub, rule, max_topk=topk, cooldown_s=cooldown_s)
    # choose_daily_topk already applies topk inside; replay_metrics also slices
    # Keep full picks (already <= topk/day).
    stats = account_stats(picks, target, position_frac)
    stats.update({"rule": rule, "state": state, "side": side, "topk": topk})
    return stats, picks


def cross_symbol_topk(
    panels: dict[str, pd.DataFrame],
    *,
    topk: int,
    cooldown_s: int,
    target: str,
    position_frac: float,
) -> tuple[dict, pd.DataFrame]:
    """Each symbol emits candidates from its primary rule; then global daily TopK."""
    cands = []
    primary = {
        "TSLA": SYMBOL_RULES["TSLA"][0],
        "NVDA": SYMBOL_RULES["NVDA"][0],
    }
    for sym, panel in panels.items():
        cfg = primary[sym]
        sub = filter_rule(panel, cfg["rule"], cfg["state"], cfg["side"])
        if sub.empty:
            continue
        # allow more local candidates then globally rank
        local = choose_daily_topk(sub, cfg["rule"], max_topk=max(5, topk), cooldown_s=cooldown_s)
        if local.empty:
            continue
        local = local.copy()
        local["symbol"] = sym
        local["active_rule"] = cfg["rule"]
        local["active_state"] = cfg["state"]
        local["edge_for_rank"] = pd.to_numeric(local[cfg["rule"]], errors="coerce").fillna(-1e9)
        cands.append(local)
    if not cands:
        return account_stats(pd.DataFrame(), target, position_frac), pd.DataFrame()
    all_c = pd.concat(cands, ignore_index=True)
    all_c = all_c.sort_values("edge_for_rank", ascending=False)
    trades = []
    for _, g in all_c.groupby("date_str", sort=False):
        last_ts = None
        chosen = 0
        # prefer diversity: at most 2 from same symbol when topk>=2
        per_sym = {}
        for r in g.itertuples(index=False):
            ts = pd.Timestamp(getattr(r, "timestamp"))
            sym = str(getattr(r, "symbol"))
            if last_ts is not None and abs((ts - last_ts).total_seconds()) <= cooldown_s:
                continue
            if topk >= 2 and per_sym.get(sym, 0) >= max(1, (topk + 1) // 2):
                continue
            trades.append(r._asdict())
            per_sym[sym] = per_sym.get(sym, 0) + 1
            last_ts = ts
            chosen += 1
            if chosen >= topk:
                break
    out = pd.DataFrame(trades)
    stats = account_stats(out, target, position_frac)
    stats.update(
        {
            "rule": "primary_per_symbol",
            "state": "mixed",
            "side": "mixed",
            "topk": topk,
            "n_nvda": int((out["symbol"] == "NVDA").sum()) if not out.empty else 0,
            "n_tsla": int((out["symbol"] == "TSLA").sum()) if not out.empty else 0,
        }
    )
    return stats, out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    target = f"target_exec_ret_{args.horizon_s}s"
    eval_months = [x.strip() for x in args.eval_months.split(",") if x.strip()]

    panels = {}
    for sym in symbols:
        print(f"[topk] prepare {sym}", flush=True)
        panel = prepare_symbol(args, sym)
        panels[sym] = panel[panel["month"].isin(eval_months)].copy()

    rows = []
    trade_frames = []

    # A) per-symbol topk
    for sym, panel in panels.items():
        for cfg in SYMBOL_RULES[sym]:
            for topk in (1, 2, 3):
                stats, picks = per_symbol_topk(
                    panel,
                    rule=cfg["rule"],
                    state=cfg["state"],
                    side=cfg["side"],
                    topk=topk,
                    cooldown_s=args.cooldown_s,
                    target=target,
                    position_frac=args.position_frac,
                )
                stats["policy"] = f"{sym}_only"
                stats["symbol"] = sym
                rows.append(stats)
                if not picks.empty:
                    p = picks.copy()
                    p["policy"] = stats["policy"]
                    p["topk"] = topk
                    p["symbol"] = sym
                    p["active_rule"] = cfg["rule"]
                    p["active_state"] = cfg["state"]
                    trade_frames.append(p)
                print(
                    f"  {sym} {cfg['rule'][:16]}|{cfg['state'][:22]} k={topk}: "
                    f"trades={stats['trades']} /day={stats['trades_per_active_day']:.2f} "
                    f"avg={stats['avg_return']:.3f} acct={stats.get('total_return_position', 0):.3f}",
                    flush=True,
                )

    # B) cross-symbol
    for topk in (1, 2, 3):
        stats, picks = cross_symbol_topk(
            panels,
            topk=topk,
            cooldown_s=args.cooldown_s,
            target=target,
            position_frac=args.position_frac,
        )
        stats["policy"] = "cross_symbol"
        stats["symbol"] = "NVDA+TSLA"
        rows.append(stats)
        if not picks.empty:
            p = picks.copy()
            p["policy"] = "cross_symbol"
            p["topk"] = topk
            trade_frames.append(p)
        print(
            f"  CROSS k={topk}: trades={stats['trades']} /day={stats['trades_per_active_day']:.2f} "
            f"avg={stats['avg_return']:.3f} acct={stats.get('total_return_position', 0):.3f} "
            f"NVDA={stats.get('n_nvda')} TSLA={stats.get('n_tsla')}",
            flush=True,
        )

    rank = pd.DataFrame(rows)
    rank.to_csv(out_dir / "policy_rank.csv", index=False)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_parquet(out_dir / "all_policy_trades.parquet", index=False)

    # NYSE-day normalized frequency
    import pandas_market_calendars as mcal

    cal = mcal.get_calendar("NYSE")
    sched = cal.schedule(start_date="2026-02-02", end_date="2026-06-30")
    nyse_days = int(len(sched))
    rank["trades_per_nyse_day"] = rank["trades"] / nyse_days

    summary = {
        "experiment": "mag7 dte0 topk + cross-symbol frequency ablation",
        "nyse_days": nyse_days,
        "symbol_rules": SYMBOL_RULES,
        "policy_rank": rank.to_dict(orient="records"),
        "files": {
            "rank": str(out_dir / "policy_rank.csv"),
            "trades": str(out_dir / "all_policy_trades.parquet"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    cols = [
        "policy",
        "symbol",
        "rule",
        "state",
        "side",
        "topk",
        "trades",
        "days",
        "trades_per_active_day",
        "trades_per_nyse_day",
        "avg_return",
        "hit_rate",
        "total_return_position",
    ]
    show = [c for c in cols if c in rank.columns]
    print(rank[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
