#!/usr/bin/env python3
"""Contract rematch on AM champ entries: ATM/OTM/DTE vs frozen ladder.

Hypothesis: foresight capture gap is partly contract choice (too far OTM /
wrong DTE), not only entry×exit. Rematch each champ entry moment under
alternate open-lock policies and re-sim TP8/SL15/h240 @20%/5.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_pocket_contract \\
    --tag research_am_pocket_contract
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

from maga7.common.config import load_profile
from maga7.common.open_lock import load_multidte_lock_index, resolve_open_lock_contract
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import to_ny
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_pocket_exit_design import _path_window, simulate_exit
from maga7.tools.scan_am_pocket_multi_gate import build_gates
from maga7.tools.scan_am_pocket_risk_optimize import (
    POCKET_SETS,
    _equity_stats,
    _month_compounds,
)
from maga7.tools.scan_session_horizon_foresight import _paths_by_ticker

DEFAULT_ENRICHED = Path(
    "/mnt/s990/data/maga7/results/research_am_pocket_multi_gate/enriched_probes.csv"
)
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")

CHAMP = "vd+cont60+mf100+volr12"
EXIT = {"mode": "tpsl", "tp": 0.08, "sl": 0.15, "max_hold": 240}


def _policies() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [
        {"name": "baseline_probe_ticker", "kind": "probe"},
    ]
    for rungs in (0, 1, 2, 3, 5):
        out.append(
            {
                "name": f"ladder_otm{rungs}_dte0",
                "kind": "resolve",
                "ladder": True,
                "otm_rungs": rungs,
                "moneyness": "ATM",
                "prefer_dte": 0,
                "allowed_dte": (0, 1, 2),
                "clear_otm_thresh": 0.01,
            }
        )
    out.append(
        {
            "name": "ladder_otm3_prefer1",
            "kind": "resolve",
            "ladder": True,
            "otm_rungs": 3,
            "moneyness": "ATM",
            "prefer_dte": 1,
            "allowed_dte": (0, 1, 2),
            "clear_otm_thresh": 0.01,
        }
    )
    out.append(
        {
            "name": "ladder_otm3_dte0_only",
            "kind": "resolve",
            "ladder": True,
            "otm_rungs": 3,
            "moneyness": "ATM",
            "prefer_dte": 0,
            "allowed_dte": (0,),
            "clear_otm_thresh": 0.01,
        }
    )
    for money in ("ATM", "OTM"):
        out.append(
            {
                "name": f"classic_{money.lower()}_dte0",
                "kind": "resolve",
                "ladder": False,
                "otm_rungs": 1,
                "moneyness": money,
                "prefer_dte": 0,
                "allowed_dte": (0, 1, 2),
                "clear_otm_thresh": 0.01,
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--enriched", default=str(DEFAULT_ENRICHED))
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--tag", default="research_am_pocket_contract")
    ap.add_argument(
        "--profile",
        default=(
            "maga7/CONFIG/strategy_profiles/"
            "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
        ),
    )
    ap.add_argument("--position-frac", type=float, default=0.20)
    ap.add_argument("--max-concurrent", type=int, default=5)
    ap.add_argument("--slip", type=float, default=0.01)
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    out = Path(prof["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    trades_root = Path(args.trades_root)
    lock = load_multidte_lock_index(Path(prof["_paths"]["open_locked_map"]).expanduser())

    probes = pd.read_csv(args.enriched)
    probes["entry_ts"] = pd.to_datetime(probes["entry_ts"])
    probes = probes[probes["enrich_ok"] == True].copy()  # noqa: E712
    pdf = pd.DataFrame(
        sorted(POCKET_SETS["no_b_up"]), columns=["session", "tod_bucket", "dir"]
    )
    probes = probes.merge(pdf, on=["session", "tod_bucket", "dir"], how="inner")

    gate = dict(build_gates())[CHAMP]
    probes = probes[probes.apply(gate, axis=1)].copy()
    print(f"champ probes={len(probes)}", flush=True)

    path_cache: dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    def paths_for(date: str, sym: str):
        key = (date, sym)
        if key not in path_cache:
            tday = load_option_trades(trades_root, sym, date)
            path_cache[key] = (
                _paths_by_ticker(tday) if tday is not None and not tday.empty else {}
            )
        return path_cache[key]

    policies = _policies()
    score_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for pol in policies:
        raw_trades: list[dict[str, Any]] = []
        n_miss = 0
        n_same = 0
        for _, r in probes.iterrows():
            date, sym = str(r["date"]), str(r["symbol"])
            et = to_ny(pd.Timestamp(r["entry_ts"]))
            direction = str(r["dir"])
            spot = float(r["spot"]) if np.isfinite(float(r.get("spot") or np.nan)) else None
            by_dte = lock.get((sym, date))
            if pol["kind"] == "probe":
                ticker = str(r["ticker"])
                dte = int(r["dte"]) if pd.notna(r.get("dte")) else None
                reason = "probe"
            else:
                ticker, dte, reason = resolve_open_lock_contract(
                    by_dte,
                    direction=direction,
                    moneyness=str(pol["moneyness"]),
                    spot=spot,
                    prefer_dte=int(pol["prefer_dte"]),
                    allowed_dte=tuple(pol["allowed_dte"]),
                    clear_otm_thresh=pol.get("clear_otm_thresh"),
                    ladder=bool(pol["ladder"]),
                    otm_rungs=int(pol["otm_rungs"]),
                )
            if not ticker:
                n_miss += 1
                continue
            if str(ticker) == str(r["ticker"]):
                n_same += 1
            arrs = paths_for(date, sym)
            path = arrs.get(str(ticker).replace("O:", ""))
            if path is None:
                n_miss += 1
                continue
            win = _path_window(path[0], path[1], et, max_hold_sec=900, slip=float(args.slip))
            if win is None:
                n_miss += 1
                continue
            rets, holds, _, _ = win
            # oracle = max ret in window (proxy; foresight used fixed horizon)
            oracle = float(np.nanmax(rets)) if len(rets) else float("nan")
            sim = simulate_exit(rets, holds, mode="tpsl", params=EXIT)
            if not np.isfinite(sim["ret"]):
                continue
            raw_trades.append(
                {
                    "date": date,
                    "symbol": sym,
                    "dir": direction,
                    "session": str(r["session"]),
                    "calendar": str(r["calendar"]),
                    "entry_ts": et,
                    "exit_ts": et + pd.Timedelta(seconds=float(sim["hold_sec"])),
                    "ret": float(sim["ret"]),
                    "exit_reason": str(sim["reason"]),
                    "hold_sec": float(sim["hold_sec"]),
                    "oracle_ret": oracle,
                    "ticker": str(ticker),
                    "dte": dte,
                    "resolve_reason": reason,
                    "same_as_probe": str(ticker) == str(r["ticker"]),
                }
            )
            detail_rows.append(
                {
                    "policy": pol["name"],
                    "date": date,
                    "symbol": sym,
                    "dir": direction,
                    "entry_ts": str(et),
                    "ticker": str(ticker),
                    "probe_ticker": str(r["ticker"]),
                    "dte": dte,
                    "ret": float(sim["ret"]),
                    "oracle_ret": oracle,
                    "same_as_probe": str(ticker) == str(r["ticker"]),
                }
            )

        disc = [t for t in raw_trades if t["calendar"] == "may_jul09"]
        blind = [t for t in raw_trades if t["calendar"] == "jul10_23"]
        sized_d = _portfolio_day(
            sorted(disc, key=lambda x: (x["entry_ts"], x["symbol"])),
            position_frac=float(args.position_frac),
            max_concurrent=int(args.max_concurrent),
            cooldown_minutes=10.0,
        )
        sized_b = _portfolio_day(
            sorted(blind, key=lambda x: (x["entry_ts"], x["symbol"])),
            position_frac=float(args.position_frac),
            max_concurrent=int(args.max_concurrent),
            cooldown_minutes=10.0,
        )
        st_d = _equity_stats(pd.DataFrame(sized_d))
        st_b = _equity_stats(pd.DataFrame(sized_b))
        months = _month_compounds(pd.DataFrame(sized_d + sized_b))
        if raw_trades:
            o = np.array([t["oracle_ret"] for t in raw_trades], dtype=float)
            rr = np.array([t["ret"] for t in raw_trades], dtype=float)
            mean_cap = float(rr.mean() / o.mean()) if o.mean() > 0 else float("nan")
            mean_oracle = float(np.nanmean(o))
            frac_same = float(np.mean([t["same_as_probe"] for t in raw_trades]))
            mean_dte = float(np.nanmean([t["dte"] for t in raw_trades if t["dte"] is not None]))
        else:
            mean_cap = mean_oracle = frac_same = mean_dte = float("nan")
        row: dict[str, Any] = {
            "policy": pol["name"],
            "n_raw": len(raw_trades),
            "n_miss": n_miss,
            "n_same_ticker": n_same,
            "frac_same_ticker": frac_same,
            "mean_dte": mean_dte,
            "mean_oracle": mean_oracle,
            "mean_capture": mean_cap,
            "may": months.get("2026-05"),
            "jun": months.get("2026-06"),
            "jul": months.get("2026-07"),
        }
        for k, v in st_d.items():
            row[f"disc_{k}"] = v
        for k, v in st_b.items():
            row[f"blind_{k}"] = v
        score_rows.append(row)
        print(
            f"{pol['name']:28s} n={len(raw_trades):3d} miss={n_miss:2d} "
            f"same={frac_same if np.isfinite(frac_same) else float('nan'):.2f} "
            f"win={float(st_d.get('trade_win') or float('nan')):.3f} "
            f"dd={float(st_d.get('maxdd') or float('nan')):.3f} "
            f"cmp={float(st_d.get('compound') or float('nan')):.3f} "
            f"cap={mean_cap if np.isfinite(mean_cap) else float('nan'):.3f} "
            f"ora={mean_oracle if np.isfinite(mean_oracle) else float('nan'):.3f}",
            flush=True,
        )

    sb = pd.DataFrame(score_rows)
    sb.to_csv(out / "scoreboard.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(out / "trade_details.csv", index=False)

    base = sb[sb.policy == "baseline_probe_ticker"]
    base_row = base.iloc[0].to_dict() if len(base) else {}
    bcmp = float(base_row.get("disc_compound") or 0)
    bcap = float(base_row.get("mean_capture") or 0)
    bdd = float(base_row.get("disc_maxdd") or -1)
    bw = float(base_row.get("disc_trade_win") or 0)

    better = sb[
        (sb["n_raw"] >= max(10, int(0.7 * int(base_row.get("n_raw") or 35))))
        & (sb["disc_trade_win"].fillna(0) >= bw - 0.05)
        & (sb["disc_maxdd"].fillna(-1) >= -0.20)
        & (
            (sb["mean_capture"].fillna(0) > bcap + 0.02)
            | (sb["disc_compound"].fillna(0) > bcmp + 0.05)
            | (sb["disc_maxdd"].fillna(-1) > bdd + 0.02)
        )
    ].sort_values(["mean_capture", "disc_compound"], ascending=[False, False])

    verdict = {
        "protocol": "contract_rematch_on_champ_entries_tp8",
        "champ_entry": CHAMP,
        "baseline": base_row,
        "better": better.head(12).to_dict(orient="records") if len(better) else [],
        "scoreboard": sb.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")

    cols = [
        c
        for c in [
            "policy",
            "n_raw",
            "frac_same_ticker",
            "mean_dte",
            "disc_trade_win",
            "disc_maxdd",
            "disc_compound",
            "mean_capture",
            "mean_oracle",
            "blind_compound",
        ]
        if c in sb.columns
    ]
    print("\nSCOREBOARD", flush=True)
    print(sb[cols].to_string(index=False), flush=True)
    print("\nBETTER than baseline", flush=True)
    print(better[cols].head(10).to_string(index=False) if len(better) else "(none)", flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
