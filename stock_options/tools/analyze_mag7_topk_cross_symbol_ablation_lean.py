#!/usr/bin/env python3
"""Lean cross-symbol TopK ablation using month-by-month processing (low memory)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from factor_lab.tools.analyze_0dte_rule_state_stability import (
    apply_rule_scorers,
    attach_all_states,
    fit_rule_scorers,
)
from factor_lab.tools.analyze_0dte_state_alpha_attribution import choose_daily_topk
from factor_lab.tools.run_0dte_adaptive_rule_pool import summarize_trades


PRIMARY = {
    "TSLA": {"rule": "tree_edge_score", "state": "is_underlying_breaking_down", "side": "CALL"},
    "NVDA": {"rule": "time_score", "state": "is_underlying_recovering", "side": "CALL"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fit-months", default="2026-02,2026-03")
    p.add_argument("--eval-months", default="2026-02,2026-03,2026-04,2026-05,2026-06")
    p.add_argument("--cooldown-s", type=int, default=30)
    p.add_argument("--horizon-s", type=int, default=30)
    p.add_argument("--position-frac", type=float, default=0.25)
    p.add_argument("--output-dir", default="stock_options/results/mag7_topk_cross_symbol_ablation_dte0")
    return p.parse_args()


def cache_path(symbol: str, month: str) -> Path:
    return Path(f"stock_options/results/mag7_state_gate_{symbol.lower()}_dte0/cache/score_dataset_{month}.parquet")


def load_fit(symbol: str, months: list[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(cache_path(symbol, m)) for m in months if cache_path(symbol, m).exists()]
    return pd.concat(frames, ignore_index=True)


def filter_rule(df: pd.DataFrame, rule: str, state: str, side: str) -> pd.DataFrame:
    out = df
    if state != "ALL":
        out = out[pd.to_numeric(out.get(state), errors="coerce").fillna(0) > 0.5]
    if side != "ALL":
        out = out[out["side"].astype(str).str.upper().eq(side)]
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fit_months = [x.strip() for x in args.fit_months.split(",") if x.strip()]
    eval_months = [x.strip() for x in args.eval_months.split(",") if x.strip()]
    target = f"target_exec_ret_{args.horizon_s}s"

    models = {}
    for sym in ("NVDA", "TSLA"):
        print(f"[lean] fit scorers {sym}", flush=True)
        fit = load_fit(sym, fit_months)
        _, weights, model = fit_rule_scorers(fit, target)
        models[sym] = (weights, model)
        del fit

    # per-symbol topk + cross candidates month by month
    per_rows = []
    cross_by_topk = {1: [], 2: [], 3: []}

    for month in eval_months:
        print(f"[lean] month {month}", flush=True)
        month_cands = []
        for sym in ("NVDA", "TSLA"):
            fp = cache_path(sym, month)
            if not fp.exists():
                continue
            df = pd.read_parquet(fp)
            df["month"] = month
            df["symbol"] = sym
            weights, model = models[sym]
            df = apply_rule_scorers(df, weights, model)
            df, _ = attach_all_states(df)
            cfg = PRIMARY[sym]
            # per-symbol topk stats
            sub = filter_rule(df, cfg["rule"], cfg["state"], cfg["side"])
            if not sub.empty:
                picks5 = choose_daily_topk(sub, cfg["rule"], max_topk=5, cooldown_s=args.cooldown_s)
                for topk in (1, 2, 3):
                    pk = picks5[pd.to_numeric(picks5["pick_rank"], errors="coerce") <= topk]
                    st = summarize_trades(pk, target, f"{sym}_k{topk}", position_frac=args.position_frac)
                    st.update({"symbol": sym, "month": month, "topk": topk, **cfg})
                    per_rows.append(st)
                # cross candidates: keep up to 5 local
                loc = picks5.copy()
                loc["edge_for_rank"] = pd.to_numeric(loc[cfg["rule"]], errors="coerce").fillna(-1e9)
                loc["active_rule"] = cfg["rule"]
                loc["active_state"] = cfg["state"]
                month_cands.append(loc)
            del df

        if not month_cands:
            continue
        all_c = pd.concat(month_cands, ignore_index=True).sort_values("edge_for_rank", ascending=False)
        for topk in (1, 2, 3):
            trades = []
            for _, g in all_c.groupby("date_str", sort=False):
                last_ts = None
                chosen = 0
                per_sym: dict[str, int] = {}
                for r in g.itertuples(index=False):
                    ts = pd.Timestamp(getattr(r, "timestamp"))
                    sym = str(getattr(r, "symbol"))
                    if last_ts is not None and abs((ts - last_ts).total_seconds()) <= args.cooldown_s:
                        continue
                    if topk >= 2 and per_sym.get(sym, 0) >= max(1, (topk + 1) // 2):
                        continue
                    trades.append(r._asdict())
                    per_sym[sym] = per_sym.get(sym, 0) + 1
                    last_ts = ts
                    chosen += 1
                    if chosen >= topk:
                        break
            if trades:
                cross_by_topk[topk].extend(trades)

    per = pd.DataFrame(per_rows)
    per.to_csv(out_dir / "per_symbol_month.csv", index=False)

    # aggregate per-symbol
    agg_rows = []
    for (sym, topk), g in per.groupby(["symbol", "topk"]):
        # reconstruct trades by summing monthly trades; avg weighted
        trades = int(g["trades"].sum())
        days = int(g["days"].sum())
        # approximate avg from monthly weighted
        wavg = float((g["avg_return"] * g["trades"]).sum() / trades) if trades else 0.0
        # account: chain monthly total_return_position approximately via sum of log - use trade-weighted rough
        # Better recompute from stored? we only have monthly summaries. Use product of (1+pos*avg)^n approx bad.
        # Re-run summarize on concatenated is better - store picks next time.
        # For now report trade counts + weighted avg/hit from monthly.
        whit = float((g["hit_rate"] * g["trades"]).sum() / trades) if trades else 0.0
        agg_rows.append(
            {
                "policy": f"{sym}_only",
                "symbol": sym,
                "topk": int(topk),
                "trades": trades,
                "active_days": days,
                "trades_per_active_day": trades / days if days else 0.0,
                "avg_return": wavg,
                "hit_rate": whit,
            }
        )

    cross_rows = []
    cross_trades_all = []
    for topk, trades in cross_by_topk.items():
        df = pd.DataFrame(trades)
        st = summarize_trades(df, target, f"cross_k{topk}", position_frac=args.position_frac) if not df.empty else {
            "trades": 0, "days": 0, "avg_return": 0.0, "hit_rate": 0.0, "total_return_position": 0.0
        }
        cross_rows.append(
            {
                "policy": "cross_symbol",
                "symbol": "NVDA+TSLA",
                "topk": topk,
                "trades": st.get("trades", 0),
                "active_days": st.get("days", 0),
                "trades_per_active_day": (st.get("trades", 0) / st.get("days", 1)) if st.get("days") else 0.0,
                "avg_return": st.get("avg_return", 0.0),
                "hit_rate": st.get("hit_rate", 0.0),
                "total_return_position": st.get("total_return_position", 0.0),
                "n_nvda": int((df["symbol"] == "NVDA").sum()) if not df.empty else 0,
                "n_tsla": int((df["symbol"] == "TSLA").sum()) if not df.empty else 0,
            }
        )
        if not df.empty:
            df = df.copy()
            df["topk"] = topk
            df["policy"] = "cross_symbol"
            cross_trades_all.append(df)

    import pandas_market_calendars as mcal

    nyse_days = len(mcal.get_calendar("NYSE").schedule("2026-02-02", "2026-06-30"))
    rank = pd.DataFrame(agg_rows + cross_rows)
    rank["trades_per_nyse_day"] = rank["trades"] / nyse_days
    rank.to_csv(out_dir / "policy_rank.csv", index=False)
    if cross_trades_all:
        pd.concat(cross_trades_all, ignore_index=True).to_parquet(out_dir / "cross_trades.parquet", index=False)

    summary = {
        "nyse_days": nyse_days,
        "primary_rules": PRIMARY,
        "policy_rank": rank.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(rank.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
