#!/usr/bin/env python3
"""Phase 3: Failure Detector ablation on real Top2 seats.

Compares baseline trail exits vs early failure EXIT. KPI:
  early-exit savings on losers > washed profit on winners;
  compound ret / MaxDD not worse on ≥2/3 OOS folds; Smooth/Impulse split.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.decision_funnel import (
    FUNNEL_VERSION,
    FROZEN_TRADE,
    FunnelConfig,
    day_decision_seats,
)
from maga7.common.failure_detector import (
    FailureDetectorConfig,
    failure_cfg_for_sleeve,
    simulate_stock_with_failure,
)
from maga7.common.replay import month_list
from maga7.common.signals import attach_mf_features, load_stock_month_files
from maga7.common.smooth_trend import SmoothStockTradeConfig
from maga7.tools.run_smooth_impulse_stock_replay import SYMS, _equity

FOLDS = [
    {"name": "fold_2025h1", "start": "2025-04-01", "end": "2025-06-30"},
    {"name": "fold_2025h2", "start": "2025-10-01", "end": "2025-12-31"},
    {"name": "fold_2026h1", "start": "2026-04-01", "end": "2026-07-17"},
]


def resolve_fd(name: str, sleeve: str) -> FailureDetectorConfig:
    s = str(sleeve).lower()
    base = failure_cfg_for_sleeve(s)
    if name == "baseline":
        return FailureDetectorConfig(enabled=False, sleeve=s)
    if name == "fd_core":
        return base
    if name == "fd_mae_only":
        return replace(
            base,
            early_giveback=9.0,
            path_min_up_frac=-1.0,
            structure_lookback=0,
            lose_open=False,
            lose_vwap=False,
        )
    if name == "fd_structure":
        return replace(
            base,
            early_mae_cut=9.0,
            early_giveback=9.0,
            path_min_up_frac=-1.0,
            lose_open=False,
            lose_vwap=False,
        )
    if name == "fd_path":
        return replace(
            base,
            early_mae_cut=9.0,
            early_giveback=9.0,
            structure_lookback=0,
            lose_open=False,
            lose_vwap=False,
        )
    if name == "fd_loose":
        return replace(base, early_mae_cut=0.007, early_giveback=0.005, max_eval_minutes=20.0)
    if name == "fd_tight":
        return replace(base, early_mae_cut=0.003, early_giveback=0.002, max_eval_minutes=10.0)
    if name == "fd_vwap":
        return replace(base, lose_vwap=True, lose_open=True)
    return base


def collect_seats(
    data: dict[str, pd.DataFrame],
    *,
    start: str,
    end: str,
) -> list[dict]:
    funnel = FunnelConfig()
    trade = {s: data[s] for s in SYMS if s in data}
    dates: set[str] = set()
    for df in trade.values():
        dates.update(df["date"].astype(str).unique().tolist())
    out: list[dict] = []
    dates_sorted = sorted(d for d in dates if start <= d <= end)
    for i, date in enumerate(dates_sorted):
        if i % 40 == 0:
            print(f"[seats] {date} ({i+1}/{len(dates_sorted)})", flush=True)
        day_by = {s: df[df["date"].astype(str) == date] for s, df in trade.items()}
        day_by = {s: d for s, d in day_by.items() if not d.empty}
        seats, _ = day_decision_seats(day_by, date=date, cfg=funnel)
        for seat in seats:
            out.append({**seat, "date": date, "_day": day_by[str(seat["symbol"]).upper()]})
    return out


def simulate_variant(
    seats: list[dict],
    *,
    variant: str,
    trade_cfg: SmoothStockTradeConfig,
) -> pd.DataFrame:
    rows: list[dict] = []
    for seat in seats:
        sleeve = str(seat["sleeve"])
        fd = resolve_fd(variant, sleeve)
        sim = simulate_stock_with_failure(
            seat["_day"],
            entry_ts=seat["detect_ts"],
            direction=seat["direction"],
            trade_cfg=trade_cfg,
            fd_cfg=fd,
            date=str(seat["date"]),
            sleeve=sleeve,
        )
        if sim is None:
            continue
        rows.append(
            {
                "date": str(seat["date"]),
                "symbol": str(seat["symbol"]).upper(),
                "direction": seat["direction"],
                "sleeve": sleeve,
                "seat_rank": int(seat["seat_rank"]),
                "detect_ts": str(seat["detect_ts"]),
                "score": float(seat["score"]),
                "variant": variant,
                **{k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in sim.items()},
            }
        )
    return pd.DataFrame(rows)


def _attr(base: pd.DataFrame, alt: pd.DataFrame) -> dict:
    if base.empty or alt.empty:
        return {}
    keys = ["date", "symbol", "direction", "detect_ts"]
    b = base.set_index(keys)
    a = alt.set_index(keys)
    common = b.index.intersection(a.index)
    if len(common) == 0:
        return {}
    br = b.loc[common, "ret"].astype(float)
    ar = a.loc[common, "ret"].astype(float)
    fired = (
        a.loc[common, "fd_fired"].astype(bool)
        if "fd_fired" in a.columns
        else pd.Series(False, index=common)
    )
    delta = ar - br
    on_losers = br < 0
    on_winners = br > 0
    return {
        "n_paired": int(len(common)),
        "n_fd_fired": int(fired.sum()),
        "fd_fire_rate": float(fired.mean()),
        "mean_delta": float(delta.mean()),
        "saved_on_losers": float(delta[on_losers].sum()) if on_losers.any() else 0.0,
        "washed_on_winners": float((-delta[on_winners]).clip(lower=0).sum()) if on_winners.any() else 0.0,
        "net_attr": float(delta.sum()),
        "mean_ret_base": float(br.mean()),
        "mean_ret_alt": float(ar.mean()),
        "win_base": float((br > 0).mean()),
        "win_alt": float((ar > 0).mean()),
    }


def _summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0, "total_ret": 0.0, "maxdd": 0.0, "win": None, "avg": None, "fd_fire_rate": 0.0}
    eq = _equity(trades, frac=0.5)
    by_sleeve = (
        trades.groupby("sleeve")
        .agg(n=("ret", "size"), mean_ret=("ret", "mean"), win=("ret", lambda s: float((s > 0).mean())))
        .reset_index()
        .to_dict(orient="records")
    )
    by_exit = trades["exit_reason"].value_counts().to_dict()
    return {
        "n": int(len(trades)),
        "total_ret": eq["total_ret"],
        "maxdd": eq["maxdd"],
        "win": eq["trade_win"],
        "avg": eq["avg_trade_ret"],
        "by_sleeve": by_sleeve,
        "by_exit": {str(k): int(v) for k, v in by_exit.items()},
        "fd_fire_rate": float(trades["fd_fired"].mean()) if "fd_fired" in trades.columns else 0.0,
    }


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
        default="/mnt/s990/data/maga7/results/failure_detector_ablation_v1",
    )
    ap.add_argument(
        "--variants",
        default="baseline,fd_core,fd_mae_only,fd_structure,fd_path,fd_loose,fd_tight",
    )
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    root = Path(prof["_paths"]["stock_root"]).expanduser()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    months = month_list(args.start_date, args.end_date)
    data: dict[str, pd.DataFrame] = {}
    for sym in SYMS:
        print(f"[load] {sym}", flush=True)
        raw = load_stock_month_files(root, sym, months)
        if raw.empty:
            continue
        raw = raw[
            (raw["date"].astype(str) >= args.start_date)
            & (raw["date"].astype(str) <= args.end_date)
        ]
        data[sym] = attach_mf_features(raw)

    trade_cfg = SmoothStockTradeConfig(
        max_hold_minutes=int(FROZEN_TRADE.max_hold_minutes),
        break_max_adverse=float(FROZEN_TRADE.break_max_adverse),
        break_min_up_frac=float(FROZEN_TRADE.break_min_up_frac),
        max_positions=2,
        first_per_symbol_dir=True,
        prefer_smooth_over_impulse=True,
    )

    seats = collect_seats(data, start=args.start_date, end=args.end_date)
    print(f"[seats] n={len(seats)}", flush=True)

    all_trades: dict[str, pd.DataFrame] = {}
    for v in variants:
        print(f"[sim] {v}", flush=True)
        tdf = simulate_variant(seats, variant=v, trade_cfg=trade_cfg)
        tdf.to_parquet(out / f"trades_{v}.parquet", index=False)
        all_trades[v] = tdf

    base = all_trades.get("baseline", pd.DataFrame())
    fold_rows = []
    variant_rows = []
    for v, tdf in all_trades.items():
        full = _summarize(tdf)
        attr = _attr(base, tdf) if v != "baseline" and not base.empty else {}
        variant_rows.append(
            {"variant": v, "window": "full", **full, **{f"attr_{k}": x for k, x in attr.items()}}
        )
        for fold in FOLDS:
            sub = tdf[(tdf["date"] >= fold["start"]) & (tdf["date"] <= fold["end"])]
            bsub = (
                base[(base["date"] >= fold["start"]) & (base["date"] <= fold["end"])]
                if not base.empty
                else pd.DataFrame()
            )
            sm = _summarize(sub)
            at = _attr(bsub, sub) if v != "baseline" and not bsub.empty else {}
            improve = None
            if v != "baseline" and not bsub.empty:
                bsm = _summarize(bsub)
                improve = bool(
                    sm["total_ret"] > bsm["total_ret"] and sm["maxdd"] >= bsm["maxdd"] - 0.005
                )
            fold_rows.append(
                {
                    "variant": v,
                    "fold": fold["name"],
                    "improve_vs_base": improve,
                    **sm,
                    **{f"attr_{k}": x for k, x in at.items()},
                }
            )

    vdf = pd.DataFrame(variant_rows)
    fdf = pd.DataFrame(fold_rows)
    vdf.to_csv(out / "summary_full.csv", index=False)
    fdf.to_csv(out / "summary_folds.csv", index=False)

    cands = vdf[vdf["variant"] != "baseline"].copy()
    best_name = None
    if not cands.empty:
        cands["_score"] = cands.get("attr_net_attr", 0).fillna(0) + cands["total_ret"].fillna(0)
        best_name = str(cands.sort_values("_score", ascending=False).iloc[0]["variant"])

    best_folds = fdf[fdf["variant"] == best_name] if best_name else pd.DataFrame()
    n_improve = int(best_folds["improve_vs_base"].fillna(False).sum()) if len(best_folds) else 0
    net_attr = 0.0
    if best_name and "attr_net_attr" in cands.columns:
        net_attr = float(cands.loc[cands.variant == best_name, "attr_net_attr"].iloc[0] or 0)

    sleeve_cmp = []
    if best_name and best_name in all_trades and not base.empty:
        for sleeve in ("smooth", "impulse"):
            b = base[base["sleeve"] == sleeve]
            a = all_trades[best_name][all_trades[best_name]["sleeve"] == sleeve]
            sleeve_cmp.append(
                {"sleeve": sleeve, "base": _summarize(b), "alt": _summarize(a), "attr": _attr(b, a)}
            )

    summary = {
        "funnel_version": FUNNEL_VERSION,
        "trade_cfg": {k: getattr(trade_cfg, k) for k in trade_cfg.__dataclass_fields__},
        "best_variant": best_name,
        "n_folds_improve": n_improve,
        "n_folds": len(FOLDS),
        "verdict": (
            "PROMOTE"
            if best_name and n_improve >= 2 and net_attr > 0
            else ("INTERESTING" if best_name and n_improve >= 1 else "NOT_USEFUL")
        ),
        "full": vdf.to_dict(orient="records"),
        "folds": fdf.to_dict(orient="records"),
        "sleeve_cmp": sleeve_cmp,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    cols_full = [
        c
        for c in (
            "variant",
            "n",
            "total_ret",
            "maxdd",
            "win",
            "avg",
            "fd_fire_rate",
            "attr_net_attr",
            "attr_saved_on_losers",
            "attr_washed_on_winners",
        )
        if c in vdf.columns
    ]
    cols_fold = [
        c
        for c in (
            "variant",
            "fold",
            "total_ret",
            "maxdd",
            "win",
            "improve_vs_base",
            "attr_net_attr",
            "fd_fire_rate",
        )
        if c in fdf.columns
    ]
    lines = [
        "# Failure Detector Ablation (Phase 3)",
        "",
        f"**Verdict: `{summary['verdict']}`** · best=`{best_name}` · "
        f"folds improve `{n_improve}/{len(FOLDS)}`",
        f"Funnel `{FUNNEL_VERSION}` · trail adverse `{trade_cfg.break_max_adverse}`",
        "",
        "## Full window",
        "",
        "```",
        vdf[cols_full].to_string(index=False),
        "```",
        "",
        "## OOS folds (improve vs baseline)",
        "",
        "```",
        fdf[cols_fold].to_string(index=False),
        "```",
        "",
        "## Sleeve split (best vs baseline)",
        "",
        "```",
        json.dumps(sleeve_cmp, indent=2, default=str),
        "```",
        "",
        "## Notes",
        "",
        "- Entries = real Top2 seats from frozen funnel (not full candidate pool).",
        "- Failure Detector only acts in early window; trail/TIME/EOD still apply.",
        "- PROMOTE requires ≥2/3 OOS folds better and positive net attribution.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {"verdict": summary["verdict"], "best_variant": best_name, "n_folds_improve": n_improve},
            indent=2,
        ),
        flush=True,
    )
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
