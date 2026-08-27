#!/usr/bin/env python3
"""OOS backtest for 3m/5m post-launch Signal Validator.

Compares, on the same candidate pool and test window:
  - enter immediately, no validator
  - wait 3m/5m, no validator (delay cost)
  - wait 3m/5m, confirm-only LGBM
  - wait 3m/5m, easy-FA-stratified confirm LGBM

Thresholds are selected on calibration only under a true-loss budget.
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
from maga7.common.signals import attach_mf_features, load_stock_month_files
from maga7.common.smooth_trend import (
    SmoothStockTradeConfig,
    _simulate_stock_path,
    apply_day_portfolio_cap,
)
from maga7.tools.bakeoff_smooth_bouncer import _train_lgbm
from maga7.tools.bakeoff_validator_next import (
    CONF_FEATS,
    _split_masks,
    easy_fa_flags,
    enrich_confirms,
)
from maga7.tools.run_smooth_impulse_stock_replay import SYMS, _equity
from maga7.tools.train_signal_validator_reject import pick_operating_point, reject_curve
from maga7.tools.train_smooth_launch_bouncer import MONTHS, _auc


def _train_variant(
    frame: pd.DataFrame,
    *,
    delay: int,
    train_mask: pd.Series,
    calib_mask: pd.Series,
    seed: int,
    stratified: bool,
    train_rank_max: int | None = None,
    calib_rank_max: int | None = None,
) -> tuple[object, list[str], np.ndarray, np.ndarray]:
    feats = [f"{c}_{delay}" for c in CONF_FEATS] + [
        c
        for c in ("tod_min", "look_ret", "path_eff", "sleeve_smooth")
        if c in frame.columns
    ]
    y_col = f"y_allow_conf_{delay}"
    tr = frame.loc[train_mask].copy()
    if train_rank_max is not None:
        tr = tr[tr["candidate_rank"] <= train_rank_max].copy()
    if stratified:
        tr = tr[
            (tr[y_col] == 1)
            | ((tr[y_col] == 0) & (tr["easy_fa"] == 1))
        ].copy()
    ca = frame.loc[calib_mask].copy()
    if calib_rank_max is not None:
        ca = ca[ca["candidate_rank"] <= calib_rank_max].copy()
    booster = _train_lgbm(
        tr[feats].astype(float).to_numpy(),
        tr[y_col].astype(int).to_numpy(),
        ca[feats].astype(float).to_numpy(),
        ca[y_col].astype(int).to_numpy(),
        feats,
        seed=seed,
    )
    return (
        booster,
        feats,
        booster.predict(ca[feats].astype(float).to_numpy()),
        ca[y_col].astype(int).to_numpy(),
    )


def _pick_threshold(
    y_calib: np.ndarray,
    p_calib: np.ndarray,
    *,
    true_loss_budget: float,
) -> dict:
    curve = reject_curve(y_calib, p_calib)
    op = pick_operating_point(curve, max_true_loss=true_loss_budget)
    if op is None:
        raise RuntimeError("empty calibration reject curve")
    return op


def _simulate_candidates(
    rows: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    *,
    delay: int,
    score: np.ndarray | None,
    threshold: float | None,
    trade_cfg: SmoothStockTradeConfig,
) -> pd.DataFrame:
    trades: list[dict] = []
    for pos, row in enumerate(rows.itertuples(index=False)):
        p = float(score[pos]) if score is not None else None
        if threshold is not None and (p is None or p <= threshold):
            continue
        raw = data.get(str(row.symbol))
        if raw is None:
            continue
        date = str(row.date)
        day = raw[raw["date"].astype(str) == date]
        if day.empty:
            continue
        entry_ts = pd.Timestamp(row.detect_ts) + pd.Timedelta(minutes=delay)
        sim = _simulate_stock_path(
            day,
            entry_ts=entry_ts,
            direction="UP",
            cfg=trade_cfg,
            date=date,
        )
        if sim is None:
            continue
        trades.append(
            {
                "date": date,
                "symbol": str(row.symbol),
                "direction": "UP",
                "sleeve": str(row.sleeve),
                "detect_ts": str(row.detect_ts),
                "score": float(row.score),
                "validator_p": p,
                "confirm_delay": delay,
                **{
                    k: (str(v) if isinstance(v, pd.Timestamp) else v)
                    for k, v in sim.items()
                },
            }
        )
    capped = apply_day_portfolio_cap(trades, max_positions=2)
    return pd.DataFrame(capped)


def _stats(label: str, trades: pd.DataFrame, **extra) -> dict:
    if trades.empty:
        eq = {
            "total_ret": 0.0,
            "maxdd": 0.0,
            "n_trades": 0,
            "trade_win": None,
            "avg_trade_ret": None,
        }
    else:
        eq = _equity(trades, frac=0.5)
    return {
        "variant": label,
        "total_ret": eq.get("total_ret"),
        "maxdd": eq.get("maxdd"),
        "n_trades": eq.get("n_trades"),
        "trade_win": eq.get("trade_win"),
        "avg_trade_ret": eq.get("avg_trade_ret"),
        **extra,
    }


def _simulate_staged(
    selected_immediate: pd.DataFrame,
    test: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    *,
    delay: int,
    score: np.ndarray,
    threshold: float,
    starter_frac: float,
    trade_cfg: SmoothStockTradeConfig,
) -> pd.DataFrame:
    """Starter now; add after confirm if passed, else flatten starter."""
    scored = test[["date", "symbol", "detect_ts", "sleeve"]].copy()
    scored["stage_p"] = score
    scored = scored.drop_duplicates(
        ["date", "symbol", "detect_ts", "sleeve"],
        keep="first",
    )
    base = selected_immediate.merge(
        scored,
        on=["date", "symbol", "detect_ts", "sleeve"],
        how="left",
        suffixes=("", "_score"),
    )
    rows: list[dict] = []
    for row in base.itertuples(index=False):
        raw = data.get(str(row.symbol))
        if raw is None:
            continue
        date = str(row.date)
        day = raw[raw["date"].astype(str) == date].copy()
        if day.empty:
            continue
        day["timestamp"] = pd.to_datetime(day["timestamp"])
        entry_ts = pd.Timestamp(row.entry_ts)
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.tz_localize("America/New_York")
        else:
            entry_ts = entry_ts.tz_convert("America/New_York")
        confirm_ts = entry_ts + pd.Timedelta(minutes=delay)
        passed = (
            pd.notna(row.stage_p)
            and float(row.stage_p) > float(threshold)
        )
        if passed:
            add_sim = _simulate_stock_path(
                day,
                entry_ts=confirm_ts,
                direction="UP",
                cfg=trade_cfg,
                date=date,
            )
            if add_sim is None:
                continue
            combined_ret = (
                float(starter_frac) * float(row.ret)
                + (1.0 - float(starter_frac)) * float(add_sim["ret"])
            )
            exit_ts = add_sim["exit_ts"]
            reason = f"PASS_ADD_{add_sim['exit_reason']}"
        else:
            d = day.sort_values("timestamp")
            before = d[d["timestamp"] >= entry_ts]
            at_confirm = d[d["timestamp"] >= confirm_ts]
            if before.empty or at_confirm.empty:
                continue
            px0 = float(before.iloc[0]["close"])
            px1 = float(at_confirm.iloc[0]["close"])
            starter_ret = px1 / px0 - 1.0 - 2.0 * (
                float(trade_cfg.cost_bps) / 1e4
            )
            combined_ret = float(starter_frac) * starter_ret
            exit_ts = at_confirm.iloc[0]["timestamp"]
            reason = "CONFIRM_FAIL_FLAT"
        rows.append(
            {
                "date": date,
                "symbol": str(row.symbol),
                "direction": "UP",
                "detect_ts": str(row.detect_ts),
                "entry_ts": str(entry_ts),
                "exit_ts": str(exit_ts),
                "ret": float(combined_ret),
                "validator_p": float(row.stage_p),
                "confirm_delay": delay,
                "starter_frac": starter_frac,
                "confirm_pass": bool(passed),
                "exit_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        default="/mnt/s990/data/maga7/results/smooth_bouncer_bakeoff_v1/dataset_up_mfe10.parquet",
    )
    ap.add_argument(
        "--profile",
        default=(
            "maga7/CONFIG/strategy_profiles/"
            "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
        ),
    )
    ap.add_argument("--train-end", default="2026-04-30")
    ap.add_argument("--calib-start", default="2026-05-01")
    ap.add_argument("--calib-end", default="2026-06-15")
    ap.add_argument("--test-start", default="2026-06-16")
    ap.add_argument("--test-end", default="2026-07-17")
    ap.add_argument("--true-loss-budget", type=float, default=0.075)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/confirm_validator_backtest_v1",
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = pd.read_parquet(args.dataset)
    ds = ds[ds["direction"].astype(str).str.upper() == "UP"].reset_index(drop=True)
    ds = easy_fa_flags(ds)

    prof = load_profile(args.profile)
    stock_root = Path(prof["_paths"]["stock_root"]).expanduser()
    data: dict[str, pd.DataFrame] = {}
    for symbol in SYMS:
        raw = load_stock_month_files(stock_root, symbol, MONTHS)
        if not raw.empty:
            data[symbol] = attach_mf_features(raw)

    conf_path = out / "dataset_confirm_3m_5m.parquet"
    if conf_path.exists():
        enriched = pd.read_parquet(conf_path)
    else:
        enriched = enrich_confirms(
            ds,
            data,
            delays=(3, 5),
            good_mfe=0.010,
            toxic_mae=0.008,
            horizon=90,
        )
        enriched = easy_fa_flags(enriched)
    enriched = enriched.sort_values(
        ["date", "detect_ts", "score"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    enriched["candidate_rank"] = (
        enriched.groupby("date", sort=False).cumcount() + 1
    )
    enriched.to_parquet(conf_path, index=False)

    train_mask, calib_mask, test_mask = _split_masks(
        enriched["date"],
        args.train_end,
        args.calib_start,
        args.calib_end,
        args.test_start,
        args.test_end,
    )
    test = enriched.loc[test_mask].copy().reset_index(drop=True)
    trade_cfg = SmoothStockTradeConfig(
        break_max_adverse=0.012,
        max_hold_minutes=180,
        break_min_up_frac=0.35,
        first_per_symbol_dir=True,
    )

    scoreboard: list[dict] = []
    immediate = _simulate_candidates(
        test,
        data,
        delay=0,
        score=None,
        threshold=None,
        trade_cfg=trade_cfg,
    )
    immediate.to_csv(out / "trades_immediate.csv", index=False)
    scoreboard.append(_stats("immediate_ungated", immediate, delay=0))

    model_meta: list[dict] = []
    for delay in (3, 5):
        delayed = _simulate_candidates(
            test,
            data,
            delay=delay,
            score=None,
            threshold=None,
            trade_cfg=trade_cfg,
        )
        delayed.to_csv(out / f"trades_wait{delay}_ungated.csv", index=False)
        scoreboard.append(_stats(f"wait{delay}_ungated", delayed, delay=delay))

        variant_specs = [
            ("full", False, None, None),
            ("strat", True, None, None),
            # Train near the actual decision funnel and calibrate on the two
            # candidates that would be selected before filtering.
            ("rank4", False, 4, 2),
        ]
        for tag, stratified, train_rank_max, calib_rank_max in variant_specs:
            booster, feats, p_calib, y_calib = _train_variant(
                enriched,
                delay=delay,
                train_mask=train_mask,
                calib_mask=calib_mask,
                seed=args.seed + delay + int(stratified),
                stratified=stratified,
                train_rank_max=train_rank_max,
                calib_rank_max=calib_rank_max,
            )
            op = _pick_threshold(
                y_calib,
                p_calib,
                true_loss_budget=float(args.true_loss_budget),
            )
            y_col = f"y_allow_conf_{delay}"
            p_test = booster.predict(test[feats].astype(float).to_numpy())
            y_test = test[y_col].astype(int).to_numpy()
            reject = p_test <= float(op["thr"])
            n_pos = max(int(y_test.sum()), 1)
            n_neg = max(int((1 - y_test).sum()), 1)
            fa_removed = float(((y_test == 0) & reject).sum() / n_neg)
            true_lost = float(((y_test == 1) & reject).sum() / n_pos)

            gated = _simulate_candidates(
                test,
                data,
                delay=delay,
                score=p_test,
                threshold=float(op["thr"]),
                trade_cfg=trade_cfg,
            )
            gated.to_csv(out / f"trades_wait{delay}_{tag}.csv", index=False)
            scoreboard.append(
                _stats(
                    f"wait{delay}_{tag}",
                    gated,
                    delay=delay,
                    threshold=float(op["thr"]),
                    fa_removed=fa_removed,
                    true_lost=true_lost,
                    reject_precision=(
                        float((y_test[reject] == 0).mean()) if reject.any() else None
                    ),
                    auc=_auc(y_test, p_test),
                    calib_fa_removed=float(op["fa_removed"]),
                    calib_true_lost=float(op["true_lost"]),
                    train_rank_max=train_rank_max,
                    calib_rank_max=calib_rank_max,
                )
            )
            model_meta.append(
                {
                    "variant": f"wait{delay}_{tag}",
                    "features": feats,
                    "threshold": float(op["thr"]),
                    "calib_op": op,
                    "test_fa_removed": fa_removed,
                    "test_true_lost": true_lost,
                    "train_rank_max": train_rank_max,
                    "calib_rank_max": calib_rank_max,
                }
            )
            for starter_frac in (0.25, 0.50):
                staged = _simulate_staged(
                    immediate,
                    test,
                    data,
                    delay=delay,
                    score=p_test,
                    threshold=float(op["thr"]),
                    starter_frac=starter_frac,
                    trade_cfg=trade_cfg,
                )
                stage_name = (
                    f"stage{delay}_{tag}_s{int(starter_frac * 100)}"
                )
                staged.to_csv(out / f"trades_{stage_name}.csv", index=False)
                scoreboard.append(
                    _stats(
                        stage_name,
                        staged,
                        delay=delay,
                        threshold=float(op["thr"]),
                        fa_removed=fa_removed,
                        true_lost=true_lost,
                        reject_precision=(
                            float((y_test[reject] == 0).mean())
                            if reject.any()
                            else None
                        ),
                        auc=_auc(y_test, p_test),
                        starter_frac=starter_frac,
                        confirm_pass_rate=(
                            float(staged["confirm_pass"].mean())
                            if not staged.empty
                            else None
                        ),
                        train_rank_max=train_rank_max,
                        calib_rank_max=calib_rank_max,
                    )
                )

    score_df = pd.DataFrame(scoreboard)
    score_df.to_csv(out / "scoreboard.csv", index=False)
    (out / "model_meta.json").write_text(
        json.dumps(model_meta, indent=2, default=str)
    )

    immediate_ret = float(
        score_df.loc[
            score_df["variant"] == "immediate_ungated", "total_ret"
        ].iloc[0]
    )
    eligible = score_df[
        score_df["variant"].str.contains("_full|_strat", regex=True)
        & (score_df["true_lost"].fillna(1.0) <= 0.10)
        & (score_df["n_trades"].fillna(0) >= 15)
    ].copy()
    winner = (
        eligible.sort_values(
            ["total_ret", "fa_removed"], ascending=False
        ).iloc[0].to_dict()
        if not eligible.empty
        else None
    )
    useful = bool(
        winner
        and float(winner["total_ret"]) >= immediate_ret + 0.005
        and float(winner["fa_removed"]) >= 0.15
    )
    summary = {
        "test_window": [args.test_start, args.test_end],
        "calib_true_loss_budget": args.true_loss_budget,
        "immediate_return": immediate_ret,
        "winner": winner,
        "useful": useful,
        "useful_rule": "ret uplift >=0.5pp, FA_removed>=15%, true_lost<=10%, n>=15",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    report = [
        "# Confirm Validator — Real Delayed-Entry Backtest",
        "",
        f"**Verdict:** `{'USEFUL' if useful else 'NOT YET USEFUL'}`",
        "",
        f"Test: {args.test_start} → {args.test_end}",
        f"Calibration true-loss budget: {args.true_loss_budget:.1%}",
        "",
        "```",
        score_df.to_string(index=False),
        "```",
        "",
        f"Winner: `{winner['variant']}`" if winner else "No eligible winner.",
        "",
        "Useful requires return uplift ≥0.5pp vs immediate, FA_removed ≥15%, "
        "true_lost ≤10%, n≥15.",
    ]
    (out / "REPORT.md").write_text("\n".join(report))
    print(score_df.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
