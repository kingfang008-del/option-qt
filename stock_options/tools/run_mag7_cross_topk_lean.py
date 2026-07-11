#!/usr/bin/env python3
"""Lean MAG7 dte0 TopK + cross-symbol ablation."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from factor_lab.tools.analyze_0dte_rule_state_stability import apply_rule_scorers, fit_rule_scorers
from factor_lab.tools.analyze_0dte_state_alpha_attribution import choose_daily_topk
from factor_lab.tools.run_0dte_adaptive_rule_pool import summarize_trades

OUT = Path("stock_options/results/mag7_topk_cross_symbol_ablation_dte0")
OUT.mkdir(parents=True, exist_ok=True)
TARGET = "target_exec_ret_30s"
FIT = ["2026-02", "2026-03"]
EVAL = ["2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
KEEP = [
    "timestamp",
    "date_str",
    "side",
    "side_code",
    "ticker",
    "bucket_id",
    TARGET,
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


def cache(sym: str, mon: str) -> Path:
    return Path(f"stock_options/results/mag7_state_gate_{sym.lower()}_dte0/cache/score_dataset_{mon}.parquet")


def load(sym: str, months: list[str], use: list[str]) -> pd.DataFrame:
    frames = []
    for m in months:
        fp = cache(sym, m)
        if not fp.exists():
            continue
        df = pd.read_parquet(fp, columns=use)
        df["month"] = m
        df["symbol"] = sym
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def add_states(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_underlying_recovering"] = (
        (pd.to_numeric(out["stock_ret_30s"], errors="coerce").fillna(0) > 0)
        & (pd.to_numeric(out["stock_ret_60s"], errors="coerce").fillna(0) < 0)
    ).astype(float)
    out["is_underlying_breaking_down"] = (
        (pd.to_numeric(out["stock_ret_30s"], errors="coerce").fillna(0) < 0)
        & (pd.to_numeric(out["stock_ret_60s"], errors="coerce").fillna(0) < 0)
        & (pd.to_numeric(out["stock_vwap_dev"], errors="coerce").fillna(0) < 0)
    ).astype(float)
    return out


def main() -> None:
    sample_cols = set(pd.read_parquet(cache("TSLA", "2026-02")).columns)
    use = [c for c in KEEP if c in sample_cols]
    print("cols", use, flush=True)

    print("fit TSLA", flush=True)
    fit = load("TSLA", FIT, use)
    fit_s = fit.sample(n=min(250000, len(fit)), random_state=42)
    _, w_tsla, m_tsla = fit_rule_scorers(fit_s, TARGET)
    del fit, fit_s

    print("fit NVDA", flush=True)
    fit = load("NVDA", FIT, use)
    fit_s = fit.sample(n=min(250000, len(fit)), random_state=42)
    _, w_nvda, m_nvda = fit_rule_scorers(fit_s, TARGET)
    del fit, fit_s

    primary = {
        "TSLA": ("tree_edge_score", "is_underlying_breaking_down", "CALL", w_tsla, m_tsla),
        "NVDA": ("time_score", "is_underlying_recovering", "CALL", w_nvda, m_nvda),
    }
    cross: dict[int, list] = {1: [], 2: [], 3: []}
    per_rows = []

    for mon in EVAL:
        print("month", mon, flush=True)
        cands = []
        for sym, (rule, state, side, w, model) in primary.items():
            fp = cache(sym, mon)
            if not fp.exists():
                continue
            df = pd.read_parquet(fp, columns=use)
            df["month"] = mon
            df["symbol"] = sym
            df = apply_rule_scorers(df, w, model)
            df = add_states(df)
            sub = df[
                (pd.to_numeric(df[state], errors="coerce").fillna(0) > 0.5)
                & (df["side"].astype(str).str.upper() == side)
            ]
            if sub.empty:
                continue
            picks = choose_daily_topk(sub, rule, max_topk=5, cooldown_s=30)
            for topk in (1, 2, 3):
                pk = picks[pd.to_numeric(picks["pick_rank"], errors="coerce") <= topk]
                st = summarize_trades(pk, TARGET, f"{sym}_k{topk}", position_frac=0.25)
                st.update(symbol=sym, month=mon, topk=topk, rule=rule, state=state)
                per_rows.append(st)
            loc = picks.copy()
            # comparable cross-symbol rank: within-symbol daily percentile
            loc["edge_for_rank"] = (
                pd.to_numeric(loc[rule], errors="coerce")
                .groupby(loc["date_str"])
                .rank(pct=True, ascending=True)
                .fillna(0.0)
            )
            loc["active_rule"] = rule
            loc["active_state"] = state
            cands.append(loc)
            del df, sub
        if not cands:
            continue
        allc = pd.concat(cands, ignore_index=True).sort_values("edge_for_rank", ascending=False)
        for topk in (1, 2, 3):
            trades = []
            for _, g in allc.groupby("date_str", sort=False):
                last = None
                n = 0
                per: dict[str, int] = {}
                for r in g.itertuples(index=False):
                    ts = pd.Timestamp(r.timestamp)
                    sym = str(r.symbol)
                    if last is not None and abs((ts - last).total_seconds()) <= 30:
                        continue
                    if topk >= 2 and per.get(sym, 0) >= max(1, (topk + 1) // 2):
                        continue
                    trades.append(r._asdict())
                    per[sym] = per.get(sym, 0) + 1
                    last = ts
                    n += 1
                    if n >= topk:
                        break
            cross[topk].extend(trades)

    per = pd.DataFrame(per_rows)
    agg = []
    for (sym, topk), g in per.groupby(["symbol", "topk"]):
        trades = int(g.trades.sum())
        days = int(g.days.sum())
        agg.append(
            dict(
                policy=f"{sym}_only",
                symbol=sym,
                topk=int(topk),
                trades=trades,
                active_days=days,
                trades_per_active_day=trades / days if days else 0,
                avg_return=float((g.avg_return * g.trades).sum() / trades) if trades else 0,
                hit_rate=float((g.hit_rate * g.trades).sum() / trades) if trades else 0,
            )
        )

    cross_rows = []
    for topk, trs in cross.items():
        df = pd.DataFrame(trs)
        st = (
            summarize_trades(df, TARGET, f"cross{topk}", position_frac=0.25)
            if len(df)
            else {"trades": 0, "days": 0, "avg_return": 0, "hit_rate": 0, "total_return_position": 0}
        )
        cross_rows.append(
            dict(
                policy="cross_symbol",
                symbol="NVDA+TSLA",
                topk=topk,
                trades=st["trades"],
                active_days=st.get("days", 0),
                trades_per_active_day=(st["trades"] / st["days"]) if st.get("days") else 0,
                avg_return=st.get("avg_return", 0),
                hit_rate=st.get("hit_rate", 0),
                total_return_position=st.get("total_return_position", 0),
                n_nvda=int((df.symbol == "NVDA").sum()) if len(df) else 0,
                n_tsla=int((df.symbol == "TSLA").sum()) if len(df) else 0,
            )
        )
        if len(df):
            df = df.copy()
            df["topk"] = topk
            df.to_parquet(OUT / f"cross_trades_k{topk}.parquet", index=False)

    nyse = len(mcal.get_calendar("NYSE").schedule("2026-02-02", "2026-06-30"))
    rank = pd.DataFrame(agg + cross_rows)
    rank["trades_per_nyse_day"] = rank["trades"] / nyse
    rank.to_csv(OUT / "policy_rank.csv", index=False)
    (OUT / "summary.json").write_text(
        json.dumps({"nyse_days": nyse, "rank": rank.to_dict("records")}, indent=2, default=str),
        encoding="utf-8",
    )
    print(rank.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("->", OUT)


if __name__ == "__main__":
    main()
