#!/usr/bin/env python3
"""Quote FillSpec dual for SMC/flow DN scout trades-PASS champions.

Structure+flow detected on stock 1s; pricing = quote FillSpec.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.config import load_profile
from maga7.common.fills import FillSpec
from maga7.common.open_lock import (
    load_multidte_lock_index,
    resolve_open_lock_contract,
    resolve_otm_rungs,
)
from maga7.common.option_quote_tpsl import entry_quote_row, simulate_quote_tpsl
from maga7.common.replay import load_quotes, path_for_ticker, to_ny
from maga7.common.smc_flow import first_smc_flow_dn_in_window, prepare_smc_flow_day
from maga7.common.stock_1s import session_dates
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_delayed_confirm_quote_dual import _ok, _prep_path, _stats
from maga7.tools.scan_session_horizon_foresight import _spot_at_arr, _stock_arrays

NY = "America/New_York"
PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
SESSIONS = (
    ("AM_0935_1030", "09:35", "10:30"),
    ("CORE_1030_1200", "10:30", "12:00"),
    ("MID_1200_1400", "12:00", "14:00"),
    ("PM_1400_1530", "14:00", "15:30"),
)
WINDOWS = (
    ("may_jul09", "2026-05-01", "2026-07-09"),
    ("jul10_23", "2026-07-10", "2026-07-23"),
)

# Trades dual-pass champion (+ near-miss flow for ablation visibility)
DEFAULT_CELLS = (
    {
        "name": "bos_disp_dn_sw300_d0.005_foff_tp0.2_sl0.15",
        "morph": "bos_disp_dn",
        "swing_sec": 300,
        "disp_sec": 60,
        "disp_thr": 0.005,
        "flow_sec": 120,
        "flow_mode": "off",
        "min_dn_vol_share": None,
        "min_streak_dn": 0,
        "require_mf_neg": False,
        "tp": 0.20,
        "sl": 0.15,
    },
    {
        "name": "bos_disp_dn_sw300_d0.005_fshare55_tp0.25_sl0.15",
        "morph": "bos_disp_dn",
        "swing_sec": 300,
        "disp_sec": 60,
        "disp_thr": 0.005,
        "flow_sec": 120,
        "flow_mode": "share55",
        "min_dn_vol_share": 0.55,
        "min_streak_dn": 0,
        "require_mf_neg": False,
        "tp": 0.25,
        "sl": 0.15,
    },
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_smc_flow_scout_dn_quote_dual")
    ap.add_argument("--max-spreads", default="0.10,0.15")
    ap.add_argument("--max-lags", default="2,3")
    ap.add_argument("--min-mid", type=float, default=0.05)
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--entry-frac", type=float, default=0.75)
    ap.add_argument("--exit-frac", type=float, default=0.75)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cooldown-minutes", type=float, default=10.0)
    ap.add_argument("--min-n", type=int, default=8)
    ap.add_argument("--min-day-win", type=float, default=0.55)
    ap.add_argument("--stride-sec", type=int, default=15)
    args = ap.parse_args(argv)

    spreads = [float(x) for x in args.max_spreads.split(",") if x.strip()]
    lags = [float(x) for x in args.max_lags.split(",") if x.strip()]
    fill = FillSpec(entry_frac=float(args.entry_frac), exit_frac=float(args.exit_frac))

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    symbols = list(prof.get("symbols") or [])
    stock_1s = Path(paths.get("stock_1s_root") or "/mnt/s990/data/raw_1s/stocks").expanduser()
    quote_root = Path(paths["quote_1s_root"])
    lock = load_multidte_lock_index(Path(paths["open_locked_map"]).expanduser())
    otm = resolve_otm_rungs(prof, default=3)
    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    start_all = min(w[1] for w in WINDOWS)
    end_all = max(w[2] for w in WINDOWS)
    dates = session_dates(start_all, end_all)
    # unique arm specs (ignore tp/sl)
    specs: dict[str, dict[str, Any]] = {}
    for c in DEFAULT_CELLS:
        key = (
            f"{c['morph']}_sw{c['swing_sec']}_d{c['disp_thr']}_f{c['flow_mode']}"
        )
        specs[key] = c

    print(
        f"smc_flow DN QUOTE dual {start_all}..{end_all} "
        f"specs={list(specs)} sp={spreads} lag={lags}",
        flush=True,
    )

    arms: list[dict[str, Any]] = []
    for di, date in enumerate(dates):
        if di % 10 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)}) arms={len(arms)}", flush=True)
        for sym in symbols:
            day = load_stock_1s_day(stock_1s, sym, date)
            arrays = prepare_smc_flow_day(day)
            if arrays is None:
                continue
            qday = _prep_path(load_quotes(quote_root, sym, date))
            if qday is None or qday.empty:
                continue
            ts_ns, px = _stock_arrays(day)
            by_dte = lock.get((sym, date))
            if not by_dte:
                continue
            for sess_name, s0, s1 in SESSIONS:
                for sk, c in specs.items():
                    warm = max(int(c["disp_sec"]), int(c["swing_sec"]), int(c["flow_sec"]))
                    t_start = pd.Timestamp(f"{date} {s0}:00", tz=NY) + pd.Timedelta(
                        seconds=warm
                    )
                    t_end = pd.Timestamp(f"{date} {s1}:00", tz=NY)
                    hit = first_smc_flow_dn_in_window(
                        arrays,
                        t_start=t_start,
                        t_end=t_end,
                        morph=str(c["morph"]),
                        swing_sec=int(c["swing_sec"]),
                        disp_sec=int(c["disp_sec"]),
                        disp_thr=float(c["disp_thr"]),
                        flow_sec=int(c["flow_sec"]),
                        min_dn_vol_share=c["min_dn_vol_share"],
                        min_streak_dn=int(c["min_streak_dn"]),
                        require_mf_neg=bool(c["require_mf_neg"]),
                        stride_sec=int(args.stride_sec),
                    )
                    if hit is None:
                        continue
                    t_arm, _arm = hit
                    spot = _spot_at_arr(ts_ns, px, t_arm)
                    ticker, dte, _ = resolve_open_lock_contract(
                        by_dte,
                        direction="DN",
                        moneyness="ATM",
                        spot=spot,
                        prefer_dte=0,
                        allowed_dte=[0, 1, 2],
                        clear_otm_thresh=0.01,
                        ladder=True,
                        otm_rungs=otm,
                    )
                    if not ticker:
                        continue
                    path = _prep_path(path_for_ticker(qday, ticker))
                    if path is None or path.empty:
                        continue
                    probe = entry_quote_row(
                        path,
                        t_arm,
                        max_lag_sec=max(lags),
                        max_spread_pct=max(spreads),
                        min_mid=float(args.min_mid),
                    )
                    if probe is None:
                        continue
                    arms.append(
                        {
                            "date": date,
                            "symbol": sym,
                            "spec_key": sk,
                            "session": sess_name,
                            "arm_ts": to_ny(t_arm),
                            "ticker": ticker,
                            "dte": dte,
                            "path": path,
                            "probe_spread": float(probe["spread_pct"]),
                            "probe_lag": float(probe["lag_sec"]),
                        }
                    )

    print(f"arms_resolvable={len(arms)}", flush=True)

    def window_of(date: str) -> str | None:
        for wname, a, b in WINDOWS:
            if a <= date <= b:
                return wname
        return None

    score_rows: list[dict[str, Any]] = []
    dual_pass: list[dict[str, Any]] = []
    trade_dump: dict[str, pd.DataFrame] = {}

    for cell in DEFAULT_CELLS:
        sk = (
            f"{cell['morph']}_sw{cell['swing_sec']}_d{cell['disp_thr']}_f{cell['flow_mode']}"
        )
        tp, sl = float(cell["tp"]), float(cell["sl"])
        for max_sp in spreads:
            for max_lag in lags:
                name = f"{cell['name']}_sp{max_sp}_lag{max_lag}"
                win_raw: dict[str, list] = {w[0]: [] for w in WINDOWS}
                n_sig = n_block = n_fill = 0
                for arm in arms:
                    if arm["spec_key"] != sk:
                        continue
                    wname = window_of(str(arm["date"]))
                    if wname is None:
                        continue
                    n_sig += 1
                    if float(arm["probe_spread"]) > max_sp or float(arm["probe_lag"]) > max_lag:
                        n_block += 1
                        continue
                    sim = simulate_quote_tpsl(
                        arm["path"],
                        arm["arm_ts"],
                        tp=tp,
                        sl=sl,
                        max_hold_sec=int(args.max_hold_sec),
                        fill=fill,
                        max_lag_sec=max_lag,
                        max_spread_pct=max_sp,
                        min_mid=float(args.min_mid),
                    )
                    if sim is None or not np.isfinite(sim["ret"]):
                        n_block += 1
                        continue
                    n_fill += 1
                    win_raw[wname].append(
                        {
                            "date": arm["date"],
                            "symbol": arm["symbol"],
                            "dir": "DN",
                            "session": arm["session"],
                            "entry_ts": str(sim["entry_ts"]),
                            "exit_ts": str(sim["exit_ts"]),
                            "ticker": arm["ticker"],
                            "ret": sim["ret"],
                            "exit_reason": sim["reason"],
                            "hold_sec": sim["hold_sec"],
                            "cell": name,
                            "event_source": "smc_flow_scout",
                            "window": wname,
                        }
                    )

                win_stats: dict[str, Any] = {}
                sized_all: list[dict] = []
                for wname, _, _ in WINDOWS:
                    raw = win_raw[wname]
                    by_d: dict[str, list] = {}
                    for r in raw:
                        by_d.setdefault(str(r["date"]), []).append(r)
                    sized: list[dict] = []
                    for _, rs in sorted(by_d.items()):
                        sized.extend(
                            _portfolio_day(
                                sorted(rs, key=lambda x: (x["entry_ts"], x["symbol"])),
                                position_frac=float(args.position_frac),
                                max_concurrent=int(args.max_concurrent),
                                cooldown_minutes=float(args.cooldown_minutes),
                            )
                        )
                    win_stats[wname] = _stats(sized)
                    sized_all.extend(sized)

                both = True
                for wname, _, _ in WINDOWS:
                    mn = int(args.min_n)
                    if wname == "jul10_23":
                        mn = min(mn, 6)
                    if not _ok(
                        win_stats[wname], min_n=mn, min_day_win=float(args.min_day_win)
                    ):
                        both = False
                        break

                row = {
                    "name": name,
                    "base": cell["name"],
                    "flow_mode": cell["flow_mode"],
                    "tp": tp,
                    "sl": sl,
                    "max_spread_pct": max_sp,
                    "max_lag_sec": max_lag,
                    "dual_pass": both,
                    "n_sig": n_sig,
                    "n_block": n_block,
                    "n_fill": n_fill,
                }
                for wname, _, _ in WINDOWS:
                    for k, v in win_stats[wname].items():
                        row[f"{wname}_{k}"] = v
                score_rows.append(row)
                if both:
                    dual_pass.append(row)
                    trade_dump[name] = pd.DataFrame(sized_all)
                    print(
                        f"  *** QUOTE DUAL PASS {name} "
                        f"MJ09 n={row.get('may_jul09_n')} mean={row.get('may_jul09_mean')} "
                        f"J10 n={row.get('jul10_23_n')} mean={row.get('jul10_23_mean')}",
                        flush=True,
                    )

    score = pd.DataFrame(score_rows)
    score.to_csv(out / "scoreboard.csv", index=False)
    dual_pass = sorted(
        dual_pass,
        key=lambda r: float(r.get("may_jul09_add") or 0) + float(r.get("jul10_23_add") or 0),
        reverse=True,
    )
    for i, p in enumerate(dual_pass[:10]):
        name = p["name"]
        if name in trade_dump and len(trade_dump[name]):
            trade_dump[name].to_csv(out / f"trades_dual{i:02d}_{name}.csv", index=False)

    summary = {
        "expert_kind": "smc_flow_scout",
        "dir": "DN",
        "pricing": "quote_FillSpec",
        "data_note": "OF proxies on stock 1s; no aggressor tape",
        "n_arms": int(len(arms)),
        "n_rows": int(len(score_rows)),
        "dual_pass_n": int(len(dual_pass)),
        "verdict": "QUOTE_PASS" if dual_pass else "QUOTE_REJECT",
        "champion": dual_pass[0] if dual_pass else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "dual_pass.json").write_text(
        json.dumps(dual_pass[:40], indent=2, default=str), encoding="utf-8"
    )
    print("\n=== verdict", summary["verdict"], "dual_pass_n=", len(dual_pass), flush=True)
    if not dual_pass and not score.empty:
        score["_sum"] = score["may_jul09_add"].fillna(0) + score["jul10_23_add"].fillna(0)
        cols = [
            c
            for c in [
                "name",
                "flow_mode",
                "may_jul09_n",
                "may_jul09_mean",
                "may_jul09_day_win",
                "jul10_23_n",
                "jul10_23_mean",
                "jul10_23_day_win",
                "n_fill",
            ]
            if c in score.columns
        ]
        print(score.sort_values("_sum", ascending=False)[cols].head(12).to_string(index=False))
    print(f"wrote {out}", flush=True)
    return 0 if dual_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
