#!/usr/bin/env python3
"""State-conditioned alpha attribution for QQQ 0DTE."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from factor_lab.tools.analyze_0dte_tradeprint_factors import load_factor_dataset
from factor_lab.tools.run_0dte_factor_score_loop import (
    SCORE_COLS,
    STATE_COLS,
    apply_ic_score,
    build_score_dataset,
    fit_ic_weights,
    fit_tree,
    predict_tree,
    spearman_ic,
)
from factor_lab.tools.run_0dte_minimal_five_layer_loop import load_stock_state_features


SCORE_NAMES = ["ic_edge_score", "tree_edge_score", "hot_score", *SCORE_COLS]
TIME_STATE_COLS = ["is_opening", "is_lunch", "is_power_hour"]


def add_time_states(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    tod = pd.to_numeric(out["timestamp"].dt.hour * 60 + out["timestamp"].dt.minute, errors="coerce")
    out["is_opening"] = ((tod >= 570) & (tod < 630)).astype(float)
    out["is_lunch"] = ((tod >= 720) & (tod < 810)).astype(float)
    out["is_power_hour"] = ((tod >= 900) & (tod < 960)).astype(float)
    return out


def add_pair_states(df: pd.DataFrame, base_states: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    pairs = []
    market_states = [c for c in base_states if c not in TIME_STATE_COLS]
    for left in market_states:
        for right in TIME_STATE_COLS:
            name = f"{left}__and__{right}"
            out[name] = (
                pd.to_numeric(out[left], errors="coerce").fillna(0.0).gt(0.5)
                & pd.to_numeric(out[right], errors="coerce").fillna(0.0).gt(0.5)
            ).astype(float)
            pairs.append(name)
    return out, pairs


def choose_daily_topk(
    df: pd.DataFrame,
    score: str,
    max_topk: int,
    cooldown_s: int,
    candidates_per_day: int = 100,
) -> pd.DataFrame:
    trades = []
    if df.empty:
        return pd.DataFrame()
    for _, g in df.groupby("date_str", sort=False):
        last_ts = None
        chosen = 0
        cand_n = max(candidates_per_day, max_topk * 20)
        candidates = g.nlargest(min(cand_n, len(g)), score)
        for row in candidates.itertuples(index=False):
            ts = pd.Timestamp(getattr(row, "timestamp"))
            if last_ts is not None and abs((ts - last_ts).total_seconds()) <= cooldown_s:
                continue
            record = row._asdict()
            record["pick_rank"] = chosen + 1
            trades.append(record)
            last_ts = ts
            chosen += 1
            if chosen >= max_topk:
                break
    return pd.DataFrame(trades)


def replay_metrics(trades: pd.DataFrame, target: str, topk: int) -> dict:
    tr = trades[pd.to_numeric(trades["pick_rank"], errors="coerce").fillna(999) <= topk].copy()
    if tr.empty:
        return {"trades": 0, "topk_per_day": int(topk)}
    r = pd.to_numeric(tr[target], errors="coerce").fillna(0.0)
    eq = (1.0 + 0.10 * r).cumprod()
    dd = eq / eq.cummax() - 1.0
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return {
        "trades": int(len(tr)),
        "days": int(tr["date_str"].nunique()),
        "topk_per_day": int(topk),
        "avg_return": float(r.mean()),
        "sum_return": float(r.sum()),
        "total_return_10pct_position": float(eq.iloc[-1] - 1.0),
        "hit_rate": float((r > 0).mean()),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "max_drawdown": float(dd.min()),
    }


def state_rows(df: pd.DataFrame, target: str, states: list[str], cooldown_s: int) -> pd.DataFrame:
    rows = []
    all_states = ["ALL", *states]
    for score in SCORE_NAMES:
        if score not in df.columns:
            continue
        for state in all_states:
            state_df = df if state == "ALL" else df[pd.to_numeric(df[state], errors="coerce").fillna(0.0) > 0.5]
            if len(state_df) < 100:
                continue
            for side in ["ALL", "CALL", "PUT"]:
                side_df = state_df if side == "ALL" else state_df[state_df["side"].eq(side)]
                if len(side_df) < 100:
                    continue
                base = {
                    "score": score,
                    "state": state,
                    "side": side,
                    "rows": int(len(side_df)),
                    "conditional_ic": spearman_ic(side_df[score], side_df[target]),
                    "base_mean": float(side_df[target].mean()),
                    "base_hit": float((side_df[target] > 0).mean()),
                }
                picks = choose_daily_topk(side_df, score, max_topk=5, cooldown_s=cooldown_s)
                for topk in (1, 2, 3, 5):
                    rows.append({**base, **replay_metrics(picks, target, topk)})
    out = pd.DataFrame(rows)
    return out


def select_states(train_matrix: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    work = train_matrix.copy()
    work = work[pd.to_numeric(work["trades"], errors="coerce").fillna(0) >= min_trades]
    work = work[pd.to_numeric(work["avg_return"], errors="coerce").fillna(-1) > 0]
    work = work[pd.to_numeric(work["profit_factor"], errors="coerce").fillna(0) > 1.05]
    work = work[pd.to_numeric(work["conditional_ic"], errors="coerce").fillna(-1) > 0]
    return work.sort_values(["avg_return", "profit_factor", "conditional_ic"], ascending=False)


def add_scores(train: pd.DataFrame, test: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    weights = fit_ic_weights(train, target)
    train = train.copy()
    test = test.copy()
    train["ic_edge_score"] = apply_ic_score(train, weights)
    test["ic_edge_score"] = apply_ic_score(test, weights)
    model = fit_tree(train, target)
    train["tree_edge_score"] = predict_tree(model, train)
    test["tree_edge_score"] = predict_tree(model, test)
    train["hot_score"] = train["score_hot_quote_tight"]
    test["hot_score"] = test["score_hot_quote_tight"]
    return train, test, weights


def load_period(args: argparse.Namespace, start: str, end: str, target: str, thresholds: dict | None):
    raw = load_factor_dataset(
        Path(args.micro_root),
        start,
        end,
        (args.horizon_s,),
        top_n=args.top_n,
        lookback_s=args.lookback_s,
        per_side=False,
        commission=args.commission_per_contract,
        max_spread_pct=args.max_spread_pct,
        min_ask=args.min_ask,
    )
    stock = load_stock_state_features(Path(args.stock_root), start, end)
    return build_score_dataset(raw, stock, target, thresholds)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--micro-root", default="/mnt/s990/data/microstructure/qqq_0dte_api_ladder")
    p.add_argument("--stock-root", default="/mnt/s990/data/raw_1s/stocks/QQQ")
    p.add_argument("--train-start", default="2026-04-13")
    p.add_argument("--train-end", default="2026-04-30")
    p.add_argument("--test-start", default="2026-05-01")
    p.add_argument("--test-end", default="2026-05-29")
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--lookback-s", type=int, default=60)
    p.add_argument("--horizon-s", type=int, default=30)
    p.add_argument("--commission-per-contract", type=float, default=0.65)
    p.add_argument("--max-spread-pct", type=float, default=0.05)
    p.add_argument("--min-ask", type=float, default=0.20)
    p.add_argument("--cooldown-s", type=int, default=30)
    p.add_argument("--min-select-trades", type=int, default=20)
    p.add_argument("--output-dir", default="factor_lab/results/0dte_state_alpha_attribution")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    target = f"target_exec_ret_{args.horizon_s}s"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[state-alpha] loading periods", flush=True)
    train, thresholds = load_period(args, args.train_start, args.train_end, target, None)
    test, _ = load_period(args, args.test_start, args.test_end, target, thresholds)
    train, test, weights = add_scores(train, test, target)
    train = add_time_states(train)
    test = add_time_states(test)
    states = [*STATE_COLS, *TIME_STATE_COLS]
    train, pair_states = add_pair_states(train, states)
    test, _ = add_pair_states(test, states)
    states = [*states, *pair_states]
    print(f"[state-alpha] train={len(train)} test={len(test)} states={len(states)}", flush=True)

    train_matrix = state_rows(train, target, states, args.cooldown_s)
    test_matrix = state_rows(test, target, states, args.cooldown_s)
    selected = select_states(train_matrix, args.min_select_trades)
    keys = ["score", "state", "side", "topk_per_day"]
    selected_oos = selected[keys].merge(test_matrix, on=keys, how="left", suffixes=("_train", "_test"))
    train_matrix.to_csv(out_dir / "train_state_matrix.csv", index=False)
    test_matrix.to_csv(out_dir / "test_state_matrix.csv", index=False)
    selected_oos.to_csv(out_dir / "selected_train_rules_oos.csv", index=False)
    summary = {
        "config": vars(args),
        "rows": {"train": int(len(train)), "test": int(len(test))},
        "ic_weights": weights,
        "train_positive_rules": int(len(selected)),
        "selected_rules_oos_positive": int((selected_oos["avg_return"] > 0).sum()) if "avg_return" in selected_oos else 0,
        "best_train_rules_oos": selected_oos.sort_values(["avg_return", "profit_factor"], ascending=False)
        .head(30)
        .to_dict("records"),
        "best_test_rules": test_matrix[test_matrix["trades"].fillna(0) >= args.min_select_trades]
        .sort_values(["avg_return", "profit_factor", "conditional_ic"], ascending=False)
        .head(30)
        .to_dict("records"),
        "files": {
            "train_state_matrix": str(out_dir / "train_state_matrix.csv"),
            "test_state_matrix": str(out_dir / "test_state_matrix.csv"),
            "selected_train_rules_oos": str(out_dir / "selected_train_rules_oos.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
