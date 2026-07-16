#!/usr/bin/env python3
"""Causal-clock uplift ablation for Mag7 open-ladder stack.

Quantifies how much can be recovered by:
1) cleaning cross-day stock gaps / stale prev_close
2) mf_window = 6/8/10
3) volume/imbalance quality filters
4) shorter post-completion decision delays (5/15/30/60s)
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import month_list, run_offline_replay
from maga7.common.signals import attach_mf_features, load_stock_month_files

PROD = (
    ROOT
    / "maga7"
    / "CONFIG"
    / "strategy_profiles"
    / "m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p20_v1.json"
)


def _nyse_days(start: str, end: str) -> list[str]:
    import pandas_market_calendars as mcal

    schedule = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    return [d.strftime("%Y-%m-%d") for d in schedule.index]


def _enrich_quality(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hl = (out["high"] - out["low"]).replace(0, np.nan)
    out["clv"] = ((out["close"] - out["low"]) / hl).fillna(0.5)
    out["range_bps"] = (hl / out["close"]).replace([np.inf, -np.inf], np.nan)
    out["vol_share_10"] = out.groupby("date")["volume"].transform(
        lambda s: s / s.rolling(10, min_periods=5).sum()
    )
    # money-flow intensity relative to recent absolute flow
    abs_net = out["net$"].abs()
    out["mf_intensity"] = abs_net / (
        out.groupby("date")["net$"]
        .transform(lambda s: s.abs().rolling(20, min_periods=5).mean())
        .replace(0, np.nan)
    )
    return out


def _load_raw_stock(profile: dict[str, Any]) -> dict[str, pd.DataFrame]:
    start = profile["date_range"]["start"]
    end = profile["date_range"]["end"]
    months = month_list(start, end)
    out: dict[str, pd.DataFrame] = {}
    for sym in profile["symbols"]:
        raw = load_stock_month_files(profile["_paths"]["stock_root"], sym, months)
        if raw.empty:
            continue
        out[sym] = raw[(raw["date"] >= start) & (raw["date"] <= end)].copy()
    return out


def _clean_gap_dates(
    raw_by: dict[str, pd.DataFrame],
    calendar: list[str],
    *,
    mode: str,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """mode: none | skip_gap_open | open_fallback"""
    prev_expected = {calendar[i]: calendar[i - 1] if i else None for i in range(len(calendar))}
    affected: set[str] = set()
    cleaned: dict[str, pd.DataFrame] = {}
    for sym, raw in raw_by.items():
        have = set(raw["date"].astype(str).unique())
        bad = {
            d
            for d in have
            if prev_expected.get(d) is not None and prev_expected[d] not in have
        }
        affected |= bad
        df = raw.copy()
        if mode == "skip_gap_open":
            df = df[~df["date"].astype(str).isin(bad)].copy()
        cleaned[sym] = df
    return cleaned, sorted(affected)


def _build_stock_by(
    raw_by: dict[str, pd.DataFrame],
    *,
    mf_window: int,
    vol_ma_window: int,
    gap_mode: str,
    calendar: list[str],
    quality: dict[str, Any] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    cleaned, affected = _clean_gap_dates(raw_by, calendar, mode=gap_mode)
    stock_by: dict[str, pd.DataFrame] = {}
    for sym, raw in cleaned.items():
        feat = attach_mf_features(
            raw,
            mf_window=mf_window,
            vol_ma_window=vol_ma_window,
        )
        if gap_mode == "open_fallback" and affected:
            for date in affected:
                mask = feat["date"].astype(str) == date
                if not mask.any():
                    continue
                op = float(feat.loc[mask, "open"].iloc[0])
                feat.loc[mask, "prev_close"] = op
                feat.loc[mask, "from_prev"] = feat.loc[mask, "close"] / op - 1.0
        feat = _enrich_quality(feat)
        if quality:
            fail = pd.Series(False, index=feat.index)
            if quality.get("vol_z_min") is not None:
                fail |= ~(feat["vol_z"] >= float(quality["vol_z_min"]))
            if quality.get("vol_share_min") is not None:
                fail |= ~(feat["vol_share_10"] >= float(quality["vol_share_min"]))
            if quality.get("mf_intensity_min") is not None:
                fail |= ~(feat["mf_intensity"] >= float(quality["mf_intensity_min"]))
            if quality.get("clv_up_min") is not None:
                # Direction-specific filters applied later via both sides being strict:
                # require extreme CLV only when combined with existing UP/DN streak logic
                # by neutralizing mediocre bars.
                mid = (feat["clv"] > float(quality.get("clv_dn_max", 0.4))) & (
                    feat["clv"] < float(quality["clv_up_min"])
                )
                fail |= mid
            # Neutralize failed bars so Rule-A cannot fire on them.
            feat.loc[fail, "vol_z"] = 0.0
            feat.loc[fail, "streak_up"] = 0
            feat.loc[fail, "streak_dn"] = 0
        stock_by[sym] = feat
    meta = {"gap_mode": gap_mode, "gap_open_dates": affected, "quality": quality or {}}
    return stock_by, meta


def _run_one(
    base_profile: dict[str, Any],
    raw_by: dict[str, pd.DataFrame],
    calendar: list[str],
    *,
    name: str,
    mf_window: int,
    delay: int,
    gap_mode: str,
    quality: dict[str, Any] | None = None,
    vol_z_min: float | None = None,
) -> dict[str, Any]:
    profile = deepcopy(base_profile)
    profile["signal"]["mf_window"] = int(mf_window)
    if vol_z_min is not None:
        profile["signal"]["vol_z_min"] = float(vol_z_min)
    profile["trade"]["bar_availability_delay_seconds"] = int(delay)
    stock_by, meta = _build_stock_by(
        raw_by,
        mf_window=int(mf_window),
        vol_ma_window=int(profile["signal"].get("vol_ma_window", 20)),
        gap_mode=gap_mode,
        calendar=calendar,
        quality=quality,
    )
    result = run_offline_replay(profile, scheme="m5_circuit", stock_by=stock_by)
    summary = result["summary"]
    row = {
        "name": name,
        "mf_window": mf_window,
        "delay_seconds": delay,
        "gap_mode": gap_mode,
        "vol_z_min": float(profile["signal"].get("vol_z_min", 1.0)),
        "quality": quality or {},
        "n_trades": summary.get("n_trades"),
        "total_ret": summary.get("total_ret"),
        "maxdd": summary.get("maxdd"),
        "trade_win": summary.get("trade_win"),
        "trade_exp": summary.get("trade_exp"),
        "end_equity": summary.get("end_equity"),
        "n_regime_block": summary.get("n_regime_block"),
        "gap_open_dates": meta["gap_open_dates"],
    }
    print(
        f"{name}: ret={row['total_ret']:+.2%} dd={row['maxdd']:.2%} "
        f"n={row['n_trades']} win={row['trade_win']:.1%}",
        flush=True,
    )
    return row


def main() -> None:
    p = argparse.ArgumentParser(description="Mag7 causal uplift ablation")
    p.add_argument("--profile", default=str(PROD))
    p.add_argument("--tag", default="causal_uplift_ablation_jan_jul")
    args = p.parse_args()

    base = load_profile(args.profile)
    out_dir = Path(base["_paths"]["results_dir"]) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    calendar = _nyse_days(base["date_range"]["start"], base["date_range"]["end"])
    print("loading stock cache...", flush=True)
    raw_by = _load_raw_stock(base)

    rows: list[dict[str, Any]] = []

    # 1) clean baseline vs dirty
    rows.append(
        _run_one(
            base,
            raw_by,
            calendar,
            name="dirty_mf10_delay60",
            mf_window=10,
            delay=60,
            gap_mode="none",
        )
    )
    rows.append(
        _run_one(
            base,
            raw_by,
            calendar,
            name="clean_skip_mf10_delay60",
            mf_window=10,
            delay=60,
            gap_mode="skip_gap_open",
        )
    )
    rows.append(
        _run_one(
            base,
            raw_by,
            calendar,
            name="clean_openfb_mf10_delay60",
            mf_window=10,
            delay=60,
            gap_mode="open_fallback",
        )
    )

    # 2) mf ablation on clean skip baseline
    for mf in (6, 8, 10):
        rows.append(
            _run_one(
                base,
                raw_by,
                calendar,
                name=f"clean_skip_mf{mf}_delay60",
                mf_window=mf,
                delay=60,
                gap_mode="skip_gap_open",
            )
        )

    # pick best mf by total_ret among clean skip mf*
    mf_rows = [r for r in rows if r["name"].startswith("clean_skip_mf") and r["delay_seconds"] == 60]
    best_mf = max(mf_rows, key=lambda r: float(r["total_ret"]))["mf_window"]

    # 3) quality filters on best mf
    quality_grid = [
        ("volz15", {"vol_z_min": 1.5}, 1.5),
        ("volz20", {"vol_z_min": 2.0}, 2.0),
        ("share08", {"vol_share_min": 0.08}, None),
        ("intensity15", {"mf_intensity_min": 1.5}, None),
        ("clv_extreme", {"clv_up_min": 0.65, "clv_dn_max": 0.35}, None),
        (
            "combo_volz15_share08",
            {"vol_z_min": 1.5, "vol_share_min": 0.08},
            1.5,
        ),
        (
            "combo_volz15_intensity15",
            {"vol_z_min": 1.5, "mf_intensity_min": 1.5},
            1.5,
        ),
        (
            "combo_all",
            {
                "vol_z_min": 1.5,
                "vol_share_min": 0.08,
                "mf_intensity_min": 1.5,
                "clv_up_min": 0.65,
                "clv_dn_max": 0.35,
            },
            1.5,
        ),
    ]
    for qname, qcfg, vz in quality_grid:
        rows.append(
            _run_one(
                base,
                raw_by,
                calendar,
                name=f"clean_skip_mf{best_mf}_delay60_{qname}",
                mf_window=best_mf,
                delay=60,
                gap_mode="skip_gap_open",
                quality=qcfg,
                vol_z_min=vz,
            )
        )

    # best quality among delay60 clean
    q_rows = [
        r
        for r in rows
        if r["name"].startswith(f"clean_skip_mf{best_mf}_delay60")
    ]
    best_q = max(q_rows, key=lambda r: (float(r["total_ret"]), -abs(float(r["maxdd"]))))

    # 4) delay curve on clean best mf, and on best quality
    for delay in (5, 15, 30, 60):
        rows.append(
            _run_one(
                base,
                raw_by,
                calendar,
                name=f"clean_skip_mf{best_mf}_delay{delay}",
                mf_window=best_mf,
                delay=delay,
                gap_mode="skip_gap_open",
            )
        )
        if best_q["quality"]:
            rows.append(
                _run_one(
                    base,
                    raw_by,
                    calendar,
                    name=f"clean_skip_mf{best_mf}_delay{delay}_bestq",
                    mf_window=best_mf,
                    delay=delay,
                    gap_mode="skip_gap_open",
                    quality=best_q["quality"],
                    vol_z_min=best_q.get("vol_z_min"),
                )
            )

    scoreboard = pd.DataFrame(rows)
    # stable json
    scoreboard_json = []
    for r in rows:
        scoreboard_json.append(
            {
                **{k: v for k, v in r.items() if k != "quality"},
                "quality": r.get("quality") or {},
            }
        )
    (out_dir / "scoreboard.json").write_text(
        json.dumps(scoreboard_json, indent=2, default=str), encoding="utf-8"
    )
    scoreboard.drop(columns=["quality", "gap_open_dates"], errors="ignore").to_csv(
        out_dir / "scoreboard.csv", index=False
    )

    dirty = next(r for r in rows if r["name"] == "dirty_mf10_delay60")
    clean = next(r for r in rows if r["name"] == "clean_skip_mf10_delay60")
    best_overall = max(
        rows,
        key=lambda r: (float(r["total_ret"]), -abs(float(r["maxdd"]))),
    )
    # Prefer delay<=60 causal configs only
    best_causal60 = max(
        [r for r in rows if int(r["delay_seconds"]) == 60 and str(r["gap_mode"]).startswith("skip")],
        key=lambda r: (float(r["total_ret"]), -abs(float(r["maxdd"]))),
    )
    summary = {
        "dirty_mf10_delay60": {
            "total_ret": dirty["total_ret"],
            "maxdd": dirty["maxdd"],
            "n_trades": dirty["n_trades"],
        },
        "clean_skip_mf10_delay60": {
            "total_ret": clean["total_ret"],
            "maxdd": clean["maxdd"],
            "n_trades": clean["n_trades"],
        },
        "best_mf_under_delay60": best_mf,
        "best_quality_under_delay60": {
            "name": best_q["name"],
            "total_ret": best_q["total_ret"],
            "maxdd": best_q["maxdd"],
            "n_trades": best_q["n_trades"],
            "quality": best_q["quality"],
        },
        "best_causal_delay60": {
            "name": best_causal60["name"],
            "total_ret": best_causal60["total_ret"],
            "maxdd": best_causal60["maxdd"],
            "n_trades": best_causal60["n_trades"],
        },
        "best_overall_including_shorter_delay": {
            "name": best_overall["name"],
            "total_ret": best_overall["total_ret"],
            "maxdd": best_overall["maxdd"],
            "n_trades": best_overall["n_trades"],
            "delay_seconds": best_overall["delay_seconds"],
        },
        "uplift_vs_dirty_delay60": {
            "clean_data": float(clean["total_ret"]) - float(dirty["total_ret"]),
            "best_delay60": float(best_causal60["total_ret"]) - float(dirty["total_ret"]),
            "best_overall": float(best_overall["total_ret"]) - float(dirty["total_ret"]),
        },
        "note": (
            "All runs use open_ladder + only_win + concurrent p20 + mf_flip. "
            "Shorter delay assumes minute-complete then wait N seconds; "
            "5/15s are optimistic live targets, not zero-delay lookahead."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"→ {out_dir}")


if __name__ == "__main__":
    main()
