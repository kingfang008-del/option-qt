#!/usr/bin/env python3
"""Ablate Mag7 regime gates (QQQ align + VIX reversal + put vixy_z) ± only_win_reenter."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import month_list, run_offline_replay
from maga7.common.signals import attach_mf_features, load_stock_month_files


def load_stock_by(profile: dict) -> dict[str, pd.DataFrame]:
    paths = profile["_paths"]
    sig = profile["signal"]
    start, end = profile["date_range"]["start"], profile["date_range"]["end"]
    months = month_list(start, end)
    out = {}
    for sym in profile["symbols"]:
        raw = load_stock_month_files(paths["stock_root"], sym, months)
        if raw.empty:
            continue
        raw = raw[(raw["date"] >= start) & (raw["date"] <= end)]
        out[sym] = attach_mf_features(
            raw,
            mf_window=int(sig.get("mf_window", 10)),
            vol_ma_window=int(sig.get("vol_ma_window", 20)),
        )
    return out


def _jul_stats(daily: pd.DataFrame) -> dict:
    all_d = daily.reset_index(drop=True)
    jul = all_d[all_d["date"].astype(str).str.startswith("2026-07")]
    mid = all_d[all_d["date"].astype(str).isin(["2026-07-07", "2026-07-08", "2026-07-09"])]
    out = {"jul_ret": None, "jul7_9": None, "d7": None, "d8": None, "d9": None}
    if jul.empty:
        return out
    i0 = jul.index[0]
    eqb = all_d.loc[i0 - 1, "equity"] if i0 > 0 else 100.0
    out["jul_ret"] = float(jul["equity"].iloc[-1] / eqb - 1.0)
    if not mid.empty:
        i1 = mid.index[0]
        eq0 = all_d.loc[i1 - 1, "equity"] if i1 > 0 else 100.0
        out["jul7_9"] = float(mid["equity"].iloc[-1] / eq0 - 1.0)
        for d, key in [("2026-07-07", "d7"), ("2026-07-08", "d8"), ("2026-07-09", "d9")]:
            sub = mid[mid["date"].astype(str) == d]
            if len(sub):
                out[key] = float(sub["day_ret"].iloc[0])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", default="2026-01-02")
    p.add_argument("--end-date", default="2026-07-13")
    p.add_argument("--scheme", default="m5_circuit", choices=["single", "m5", "m5_circuit"])
    p.add_argument("--tag", default="regime_ablation_jan_jul")
    args = p.parse_args()

    profile = load_profile()
    profile["date_range"]["start"] = args.start_date
    profile["date_range"]["end"] = args.end_date
    stock_by = load_stock_by(profile)

    base_reg = {
        "enabled": True,
        "qqq_align": True,
        "qqq_from_prev_eps": 0.0,
        "vix_reversal_max": 6,
        "vix_reversal_window": 30,
        "vix_reversal_pct": 0.0015,
        "put_vixy_z_min": 0.25,
        "block_on_missing": False,
    }

    variants = {
        "baseline": {"regime": {"enabled": False}, "trade": {}},
        "only_win": {"regime": {"enabled": False}, "trade": {"only_reenter_after_win": True}},
        "regime_full": {"regime": dict(base_reg), "trade": {}},
        "regime_qqq_only": {
            "regime": {**base_reg, "vix_reversal_max": None, "put_vixy_z_min": None},
            "trade": {},
        },
        "regime_vix_only": {
            "regime": {**base_reg, "qqq_align": False, "put_vixy_z_min": None},
            "trade": {},
        },
        "regime_put_only": {
            "regime": {**base_reg, "qqq_align": False, "vix_reversal_max": None},
            "trade": {},
        },
        "regime+only_win": {
            "regime": dict(base_reg),
            "trade": {"only_reenter_after_win": True},
        },
        "single_no_regime": {"regime": {"enabled": False}, "trade": {}, "scheme": "single"},
    }

    print("building regime frame once…")
    from maga7.common.regime import Mag7RegimeGate, build_regime_frame

    months = month_list(args.start_date, args.end_date)
    reg_frame = build_regime_frame(
        profile["_paths"]["stock_root"],
        months,
        start=args.start_date,
        end=args.end_date,
    )
    print(f"regime bars={len(reg_frame)}")

    out_dir = Path(profile["_paths"]["results_dir"]) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, patch in variants.items():
        cfg = copy.deepcopy(profile)
        cfg["regime"] = patch.get("regime") or {"enabled": False}
        cfg["trade"].update(patch.get("trade") or {})
        scheme = patch.get("scheme") or args.scheme
        gate = None
        if cfg["regime"].get("enabled") and not reg_frame.empty:
            gate = Mag7RegimeGate(frame=reg_frame, cfg=cfg["regime"])
        r = run_offline_replay(cfg, scheme=scheme, stock_by=stock_by, regime_gate=gate)
        s = r["summary"]
        js = _jul_stats(r["daily"])
        row = {"name": name, "scheme": scheme, **s, **js}
        rows.append(row)
        print(
            f"{name:18} ret={s['total_ret']*100:+7.0f}% dd={s['maxdd']*100:6.1f}% "
            f"n={s['n_trades']:4d} block={s.get('n_regime_block', 0):4d} "
            f"jul={((js['jul_ret'] or 0)*100):+6.1f}% mid={((js['jul7_9'] or 0)*100):+6.1f}% "
            f"d8={((js['d8'] or 0)*100):+.1f}%"
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "scoreboard.csv", index=False)
    (out_dir / "scoreboard.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
