#!/usr/bin/env python3
"""Phase 1: build real Top2 decision-seat dataset (2024–2026).

Replays frozen funnel (Smooth∨Impulse → Top2 → replacement chain),
labels each seat with ATR-normalized first-passage, and attaches
causal entry features. Sample unit = real decision seat, not full pool.
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
from maga7.common.decision_funnel import (
    FUNNEL_VERSION,
    FunnelConfig,
    SYMS_MAG7,
    day_decision_seats,
)
from maga7.common.first_passage import FirstPassageConfig, first_passage_label
from maga7.common.lgbm_bouncer import extract_bouncer_features
from maga7.common.replay import month_list
from maga7.common.signals import attach_mf_features, load_stock_month_files

NY = "America/New_York"


def _multiscale_path_feats(day: pd.DataFrame, *, asof_ts, direction: str) -> dict[str, float]:
    """Causal multi-scale path stats (3/5/10/20m) at decision time."""
    d = day.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    if getattr(d["timestamp"].dt, "tz", None) is None:
        d["timestamp"] = d["timestamp"].dt.tz_localize(NY)
    else:
        d["timestamp"] = d["timestamp"].dt.tz_convert(NY)
    asof = pd.Timestamp(asof_ts)
    if asof.tzinfo is None:
        asof = asof.tz_localize(NY)
    else:
        asof = asof.tz_convert(NY)
    upto = d[d["timestamp"] <= asof]
    if upto.empty:
        return {}
    px = float(upto.iloc[-1]["close"])
    if px <= 0:
        return {}
    out: dict[str, float] = {}
    side = str(direction).upper()
    for w in (3, 5, 10, 20):
        win = upto[upto["timestamp"] >= asof - pd.Timedelta(minutes=w)]
        if len(win) < 3:
            out[f"ms{w}_ret"] = 0.0
            out[f"ms{w}_path_eff"] = 0.0
            out[f"ms{w}_max_dd"] = 0.0
            continue
        cl = pd.to_numeric(win["close"], errors="coerce").to_numpy(dtype=float)
        net = cl[-1] / cl[0] - 1.0
        if side == "DN":
            net = -net
        path = float(np.abs(np.diff(cl) / cl[:-1]).sum()) if len(cl) > 1 else 0.0
        pe = float(abs(cl[-1] / cl[0] - 1.0) / path) if path > 1e-12 else 0.0
        if side == "UP":
            peak = np.maximum.accumulate(cl)
            dd = float(((cl / peak) - 1.0).min())
        else:
            trough = np.minimum.accumulate(cl)
            dd = float(-((cl / trough) - 1.0).max())
        out[f"ms{w}_ret"] = float(net)
        out[f"ms{w}_path_eff"] = float(pe)
        out[f"ms{w}_max_dd"] = float(dd)
    return out


def _load_universe(
    stock_root: Path,
    *,
    start: str,
    end: str,
    symbols: list[str],
) -> dict[str, pd.DataFrame]:
    months = month_list(start, end)
    out: dict[str, pd.DataFrame] = {}
    for sym in list(dict.fromkeys(symbols + ["QQQ"])):
        print(f"[load] {sym} months={len(months)}", flush=True)
        raw = load_stock_month_files(stock_root, sym, months)
        if raw.empty:
            print(f"[warn] empty {sym}", flush=True)
            continue
        raw = raw[(raw["date"].astype(str) >= start) & (raw["date"].astype(str) <= end)].copy()
        out[sym] = attach_mf_features(raw)
    return out


def build_dataset(
    data: dict[str, pd.DataFrame],
    *,
    start: str,
    end: str,
    funnel_cfg: FunnelConfig | None = None,
    label_cfg: FirstPassageConfig | None = None,
    symbols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    funnel_cfg = funnel_cfg or FunnelConfig()
    label_cfg = label_cfg or FirstPassageConfig()
    symbols = symbols or list(SYMS_MAG7)
    trade = {s: data[s] for s in symbols if s in data}
    qqq = data.get("QQQ")

    dates: set[str] = set()
    for df in trade.values():
        dates.update(df["date"].astype(str).unique().tolist())
    dates_sorted = sorted(d for d in dates if start <= d <= end)

    seat_rows: list[dict] = []
    cand_rows: list[dict] = []
    alt_rows: list[dict] = []

    for i, date in enumerate(dates_sorted):
        if i % 40 == 0:
            print(f"[day] {date} ({i+1}/{len(dates_sorted)})", flush=True)
        day_by = {s: df[df["date"].astype(str) == date] for s, df in trade.items()}
        day_by = {s: d for s, d in day_by.items() if not d.empty}
        if len(day_by) < 2:
            continue
        seats, cands = day_decision_seats(day_by, date=date, cfg=funnel_cfg)
        for c in cands:
            cand_rows.append(
                {
                    "date": date,
                    "symbol": c["symbol"],
                    "direction": c["direction"],
                    "sleeve": c["sleeve"],
                    "detect_ts": str(c["detect_ts"]),
                    "decision_ts": str(c["detect_ts"]),
                    "available_ts": str(c["detect_ts"]),
                    "event_ts": str(c["detect_ts"]),
                    "score": c["score"],
                    "look_ret": c["look_ret"],
                    "path_eff": c["path_eff"],
                    "up_frac": c["up_frac"],
                    "max_dd": c["max_dd"],
                    "from_extreme": c["from_extreme"],
                    "price": c["price"],
                    "funnel_version": FUNNEL_VERSION,
                }
            )
        if not seats:
            continue
        qday = qqq[qqq["date"].astype(str) == date] if qqq is not None and not qqq.empty else None
        for seat in seats:
            sym = str(seat["symbol"]).upper()
            day = day_by.get(sym)
            if day is None or day.empty:
                continue
            # Prior sessions for causal daily ATR (need completed days).
            sym_hist = trade[sym]
            atr_hist = sym_hist[sym_hist["date"].astype(str) <= date].tail(12000)
            lab = first_passage_label(
                day,
                entry_ts=seat["detect_ts"],
                direction=seat["direction"],
                date=date,
                cfg=label_cfg,
                atr_hist=atr_hist,
            )
            if lab is None:
                continue
            feat = extract_bouncer_features(
                symbol=sym,
                direction=seat["direction"],
                asof_ts=seat["detect_ts"],
                stock_df=day,
                qqq_df=qday,
            ) or {}
            ms = _multiscale_path_feats(day, asof_ts=seat["detect_ts"], direction=seat["direction"])
            chain = seat.get("replacement_chain") or []
            row = {
                "date": date,
                "symbol": sym,
                "direction": seat["direction"],
                "sleeve": seat["sleeve"],
                "seat_rank": int(seat["seat_rank"]),
                "is_selected": True,
                "detect_ts": str(seat["detect_ts"]),
                "decision_ts": str(seat["detect_ts"]),
                "available_ts": str(seat["detect_ts"]),
                "event_ts": str(seat["detect_ts"]),
                "price": float(seat["price"]),
                "score": float(seat["score"]),
                "look_ret": float(seat["look_ret"]),
                "path_eff": float(seat["path_eff"]),
                "up_frac": float(seat["up_frac"]),
                "max_dd": float(seat["max_dd"]),
                "from_extreme": float(seat["from_extreme"]),
                "n_day_candidates": int(len(cands)),
                "n_alts": int(len(chain)),
                "alt0_symbol": chain[0]["symbol"] if chain else None,
                "alt0_direction": chain[0]["direction"] if chain else None,
                "alt0_detect_ts": str(chain[0]["detect_ts"]) if chain else None,
                "alt0_score": float(chain[0]["score"]) if chain else None,
                "funnel_version": FUNNEL_VERSION,
                **lab,
                **feat,
                **ms,
            }
            seat_rows.append(row)
            for a in chain:
                alt_rows.append(
                    {
                        "date": date,
                        "seat_symbol": sym,
                        "seat_rank": int(seat["seat_rank"]),
                        "seat_detect_ts": str(seat["detect_ts"]),
                        "alt_rank": int(a["alt_rank"]),
                        "symbol": a["symbol"],
                        "direction": a["direction"],
                        "sleeve": a["sleeve"],
                        "detect_ts": str(a["detect_ts"]),
                        "score": float(a["score"]),
                        "look_ret": float(a["look_ret"]),
                        "funnel_version": FUNNEL_VERSION,
                    }
                )

    seats_df = pd.DataFrame(seat_rows)
    cands_df = pd.DataFrame(cand_rows)
    alts_df = pd.DataFrame(alt_rows)
    return seats_df, cands_df, alts_df


def _coverage_report(seats: pd.DataFrame, cands: pd.DataFrame, alts: pd.DataFrame) -> dict:
    if seats.empty:
        return {"n_seats": 0}
    seats = seats.copy()
    seats["year"] = seats["date"].astype(str).str.slice(0, 4)
    lab_atr = seats["label_atr"].value_counts(dropna=False).to_dict()
    lab_pct = seats["label_pct"].value_counts(dropna=False).to_dict()
    by_year = (
        seats.groupby("year")
        .agg(
            n_seats=("symbol", "size"),
            clear_true_atr=("y_clear_true_atr", "mean"),
            clear_false_atr=("y_clear_false_atr", "mean"),
            clear_true_pct=("y_clear_true_pct", "mean"),
            clear_false_pct=("y_clear_false_pct", "mean"),
            with_alt=("n_alts", lambda s: float((s > 0).mean())),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    by_sym = (
        seats.groupby("symbol")
        .agg(
            n=("symbol", "size"),
            clear_true_atr=("y_clear_true_atr", "mean"),
            clear_false_atr=("y_clear_false_atr", "mean"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
        .to_dict(orient="records")
    )
    by_sleeve = (
        seats.groupby("sleeve")
        .agg(
            n=("symbol", "size"),
            clear_true_atr=("y_clear_true_atr", "mean"),
            clear_false_atr=("y_clear_false_atr", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    train_atr = seats["y_train_atr"].dropna()
    train_pct = seats["y_train_pct"].dropna()
    return {
        "funnel_version": FUNNEL_VERSION,
        "n_seats": int(len(seats)),
        "n_days": int(seats["date"].nunique()),
        "n_candidates": int(len(cands)),
        "n_alts": int(len(alts)),
        "seats_per_day": float(len(seats) / max(seats["date"].nunique(), 1)),
        "date_min": str(seats["date"].min()),
        "date_max": str(seats["date"].max()),
        "label_atr_counts": {str(k): int(v) for k, v in lab_atr.items()},
        "label_pct_counts": {str(k): int(v) for k, v in lab_pct.items()},
        "clear_true_atr_rate": float(seats["y_clear_true_atr"].mean()),
        "clear_false_atr_rate": float(seats["y_clear_false_atr"].mean()),
        "ambiguous_atr_rate": float((seats["label_atr"] == "ambiguous").mean()),
        "n_train_atr": int(len(train_atr)),
        "train_atr_pos_rate": float(train_atr.mean()) if len(train_atr) else None,
        "n_train_pct": int(len(train_pct)),
        "train_pct_pos_rate": float(train_pct.mean()) if len(train_pct) else None,
        "with_replacement_rate": float((seats["n_alts"] > 0).mean()),
        "by_year": by_year,
        "by_symbol": by_sym,
        "by_sleeve": by_sleeve,
    }


def _write_report(summary: dict, out: Path) -> None:
    lines = [
        "# Top2 Decision Dataset (Phase 1)",
        "",
        f"**Funnel:** `{summary.get('funnel_version')}`",
        f"**Seats:** `{summary.get('n_seats')}` over `{summary.get('n_days')}` days "
        f"(`{summary.get('date_min')}` → `{summary.get('date_max')}`)",
        f"**Candidates:** `{summary.get('n_candidates')}` · "
        f"**alts rows:** `{summary.get('n_alts')}`",
        "",
        "## Labels (ATR-normalized first-passage)",
        "",
        f"- clear_true rate: `{summary.get('clear_true_atr_rate'):.3f}`",
        f"- clear_false rate: `{summary.get('clear_false_atr_rate'):.3f}`",
        f"- ambiguous rate: `{summary.get('ambiguous_atr_rate'):.3f}`",
        f"- train (non-ambiguous) n / pos: "
        f"`{summary.get('n_train_atr')}` / `{summary.get('train_atr_pos_rate')}`",
        "",
        "```",
        json.dumps(summary.get("label_atr_counts"), indent=2),
        "```",
        "",
        "## By year",
        "",
        "```",
        pd.DataFrame(summary.get("by_year") or []).to_string(index=False),
        "```",
        "",
        "## By symbol",
        "",
        "```",
        pd.DataFrame(summary.get("by_symbol") or []).to_string(index=False),
        "```",
        "",
        "## By sleeve",
        "",
        "```",
        pd.DataFrame(summary.get("by_sleeve") or []).to_string(index=False),
        "```",
        "",
        "## Notes",
        "",
        "- Sample unit = real Top2 seat after frozen funnel.",
        "- Primary train label: `y_train_atr` (clear_true=1, clear_false=0; ambiguous dropped).",
        "- `label_pct` retained as fixed-% baseline.",
        "- Replacement chain in `alts.parquet` / seat `alt0_*` columns.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default=(
            "maga7/CONFIG/strategy_profiles/"
            "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
        ),
    )
    ap.add_argument("--start-date", default="2024-01-01")
    ap.add_argument("--end-date", default="2026-07-17")
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/top2_decision_dataset_v1",
    )
    ap.add_argument("--horizon-minutes", type=int, default=90)
    ap.add_argument("--good-mfe-pct", type=float, default=0.01)
    ap.add_argument("--toxic-mae-pct", type=float, default=0.005)
    ap.add_argument("--atr-days", type=int, default=14)
    ap.add_argument("--atr-window", type=int, default=60)
    ap.add_argument("--good-mfe-atr", type=float, default=0.50)
    ap.add_argument("--toxic-mae-atr", type=float, default=0.25)
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    stock_root = Path(prof["_paths"]["stock_root"]).expanduser()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    funnel_cfg = FunnelConfig()
    label_cfg = FirstPassageConfig(
        horizon_minutes=int(args.horizon_minutes),
        good_mfe_pct=float(args.good_mfe_pct),
        toxic_mae_pct=float(args.toxic_mae_pct),
        atr_days=int(args.atr_days),
        atr_window=int(args.atr_window),
        good_mfe_atr=float(args.good_mfe_atr),
        toxic_mae_atr=float(args.toxic_mae_atr),
    )

    data = _load_universe(
        stock_root,
        start=args.start_date,
        end=args.end_date,
        symbols=list(SYMS_MAG7),
    )
    seats, cands, alts = build_dataset(
        data,
        start=args.start_date,
        end=args.end_date,
        funnel_cfg=funnel_cfg,
        label_cfg=label_cfg,
    )
    if seats.empty:
        raise SystemExit("no seats produced")

    seats.to_parquet(out / "seats.parquet", index=False)
    cands.to_parquet(out / "candidates.parquet", index=False)
    alts.to_parquet(out / "alts.parquet", index=False)

    summary = _coverage_report(seats, cands, alts)
    summary["label_cfg"] = {
        "horizon_minutes": label_cfg.horizon_minutes,
        "good_mfe_pct": label_cfg.good_mfe_pct,
        "toxic_mae_pct": label_cfg.toxic_mae_pct,
        "good_mfe_atr": label_cfg.good_mfe_atr,
        "toxic_mae_atr": label_cfg.toxic_mae_atr,
        "atr_days": label_cfg.atr_days,
        "atr_window": label_cfg.atr_window,
    }
    summary["funnel"] = funnel_cfg.to_dict()
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _write_report(summary, out)
    print(json.dumps({k: summary[k] for k in (
        "n_seats", "n_days", "n_candidates", "clear_true_atr_rate",
        "clear_false_atr_rate", "ambiguous_atr_rate", "n_train_atr",
        "train_atr_pos_rate", "with_replacement_rate",
    )}, indent=2), flush=True)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
