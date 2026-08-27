#!/usr/bin/env python3
"""AM pocket offline acceptance on option *trade prints* (no 09:30 quotes).

Historical open-lock 1s quotes lack 09:30–10:00 NBBO for most champ contracts.
Offline gate therefore uses trade-last first-passage TP/SL (slip), same dual
windows as pulse trades dual. Quote FillSpec remains a live/IB concern.

Freeze cell:
  no_b_up + vd+cont60+mf100+volr12 + TP8/SL15/h240 @20%/max5

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_pocket_trades_dual \\
    --tag research_am_pocket_trades_dual
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
from maga7.common.option_trade_tpsl import simulate_trade_tpsl
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import to_ny
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_delayed_confirm_quote_dual import _ok, _stats
from maga7.tools.scan_am_pocket_multi_gate import build_gates
from maga7.tools.scan_am_pocket_risk_optimize import POCKET_SETS, _equity_stats, _month_compounds
from maga7.tools.scan_session_horizon_foresight import _paths_by_ticker

DEFAULT_ENRICHED = Path(
    "/mnt/s990/data/maga7/results/research_am_pocket_multi_gate/enriched_probes.csv"
)
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")
PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
WINDOWS = (
    ("may_jul09", "2026-05-01", "2026-07-09"),
    ("jul10_23", "2026-07-10", "2026-07-23"),
)
NY = "America/New_York"

# Cells: champ freeze + nearby controls
CELLS: list[dict[str, Any]] = [
    {
        "name": "champ_tp8_sl15_h240",
        "entry": "vd+cont60+mf100+volr12",
        "tp": 0.08,
        "sl": 0.15,
        "max_hold": 240,
        "freeze": True,
    },
    {
        "name": "vd_soft_tp8_sl15_h240",
        "entry": "vd_soft",
        "tp": 0.08,
        "sl": 0.15,
        "max_hold": 240,
        "freeze": False,
    },
    {
        "name": "vd_volr12_tp8_sl15_h240",
        "entry": "vd+volr12",
        "tp": 0.08,
        "sl": 0.15,
        "max_hold": 240,
        "freeze": False,
    },
    {
        "name": "champ_tp10_sl15_h300",
        "entry": "vd+cont60+mf100+volr12",
        "tp": 0.10,
        "sl": 0.15,
        "max_hold": 300,
        "freeze": False,
    },
    {
        "name": "champ_tp8_sl12_h240",
        "entry": "vd+cont60+mf100+volr12",
        "tp": 0.08,
        "sl": 0.12,
        "max_hold": 240,
        "freeze": False,
    },
]


def _window_of(date: str) -> str | None:
    for name, a, b in WINDOWS:
        if a <= date <= b:
            return name
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--enriched", default=str(DEFAULT_ENRICHED))
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--tag", default="research_am_pocket_trades_dual")
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--position-frac", type=float, default=0.20)
    ap.add_argument("--max-concurrent", type=int, default=5)
    ap.add_argument("--cooldown-minutes", type=float, default=10.0)
    ap.add_argument("--min-n", type=int, default=8)
    ap.add_argument("--min-day-win", type=float, default=0.55)
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    out = Path(prof["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    trades_root = Path(args.trades_root)

    probes = pd.read_csv(args.enriched)
    probes["entry_ts"] = pd.to_datetime(probes["entry_ts"])
    probes = probes[probes["enrich_ok"] == True].copy()  # noqa: E712
    pdf = pd.DataFrame(sorted(POCKET_SETS["no_b_up"]), columns=["session", "tod_bucket", "dir"])
    probes = probes.merge(pdf, on=["session", "tod_bucket", "dir"], how="inner")
    print(f"pocket probes={len(probes)}", flush=True)

    gate_map = dict(build_gates())
    tcache: dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    def paths_for(date: str, sym: str):
        key = (date, sym)
        if key not in tcache:
            tday = load_option_trades(trades_root, sym, date)
            tcache[key] = (
                _paths_by_ticker(tday) if tday is not None and not tday.empty else {}
            )
        return tcache[key]

    # Preload path arrays per probe row
    base_rows: list[dict[str, Any]] = []
    for _, r in probes.iterrows():
        date, sym = str(r["date"]), str(r["symbol"])
        w = _window_of(date)
        if w is None:
            continue
        arr = paths_for(date, sym).get(str(r["ticker"]).replace("O:", ""))
        if arr is None:
            continue
        base_rows.append(
            {
                "row": r,
                "date": date,
                "symbol": sym,
                "dir": str(r["dir"]),
                "session": str(r["session"]),
                "ticker": str(r["ticker"]).replace("O:", ""),
                "window": w,
                "entry_ts": to_ny(pd.Timestamp(r["entry_ts"])),
                "pts": arr[0],
                "plast": arr[1],
                "oracle_ret": float(r["oracle_ret"]),
            }
        )
    print(f"with_trade_path={len(base_rows)}", flush=True)

    score_rows: list[dict[str, Any]] = []
    dual_pass: list[dict[str, Any]] = []
    trade_dumps: dict[str, pd.DataFrame] = {}

    for cell in CELLS:
        gfn = gate_map[str(cell["entry"])]
        win_raw: dict[str, list[dict[str, Any]]] = {w[0]: [] for w in WINDOWS}
        for p in base_rows:
            if not gfn(p["row"]):
                continue
            sim = simulate_trade_tpsl(
                p["pts"],
                p["plast"],
                p["entry_ts"],
                tp=float(cell["tp"]),
                sl=float(cell["sl"]),
                max_hold_sec=int(cell["max_hold"]),
                slip=float(args.slip),
            )
            if sim is None or not np.isfinite(sim["ret"]):
                continue
            et = p["entry_ts"]
            win_raw[p["window"]].append(
                {
                    "date": p["date"],
                    "symbol": p["symbol"],
                    "dir": p["dir"],
                    "session": p["session"],
                    "entry_ts": et,
                    "exit_ts": et + pd.Timedelta(seconds=float(sim["hold_sec"])),
                    "ticker": p["ticker"],
                    "ret": float(sim["ret"]),
                    "exit_reason": str(sim["reason"]),
                    "hold_sec": float(sim["hold_sec"]),
                    "oracle_ret": float(p["oracle_ret"]),
                    "cell": cell["name"],
                    "window": p["window"],
                    "event_source": "am_pocket_sleeve",
                }
            )

        win_stats: dict[str, dict[str, Any]] = {}
        sized_all: list[dict[str, Any]] = []
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
            st = _stats(sized)
            st_eq = _equity_stats(pd.DataFrame(sized)) if sized else {}
            for k, v in st_eq.items():
                st[f"eq_{k}"] = v
            win_stats[wname] = st
            sized_all.extend(sized)

        both_pulse = True
        for wname, _, _ in WINDOWS:
            mn = int(args.min_n)
            if wname == "jul10_23":
                mn = min(mn, 6)
            if not _ok(win_stats[wname], min_n=mn, min_day_win=float(args.min_day_win)):
                both_pulse = False
                break

        # Pocket research dual: both windows equity compound > 0 (blind n can be thin).
        disc_c = float(win_stats["may_jul09"].get("eq_compound") or 0)
        blind_c = float(win_stats["jul10_23"].get("eq_compound") or 0)
        n_d = int(win_stats["may_jul09"].get("n") or 0)
        n_b = int(win_stats["jul10_23"].get("n") or 0)
        both_econ = bool(n_d >= 8 and n_b >= 3 and disc_c > 0 and blind_c > 0)

        months = _month_compounds(pd.DataFrame(sized_all)) if sized_all else {}
        if sized_all:
            o = np.array([t["oracle_ret"] for t in sized_all], dtype=float)
            rr = np.array([t["ret"] for t in sized_all], dtype=float)
            mean_cap = float(rr.mean() / o.mean()) if np.nanmean(o) > 0 else float("nan")
        else:
            mean_cap = float("nan")

        row: dict[str, Any] = {
            **{k: cell[k] for k in ("name", "entry", "tp", "sl", "max_hold", "freeze")},
            "pulse_dual_pass": both_pulse,
            "econ_dual_pass": both_econ,
            "dual_pass": both_econ,  # official offline gate for pocket (trade-mark)
            "n_raw": sum(len(v) for v in win_raw.values()),
            "mean_capture": mean_cap,
            "may": months.get("2026-05"),
            "jun": months.get("2026-06"),
            "jul": months.get("2026-07"),
        }
        for wname, _, _ in WINDOWS:
            for k, v in win_stats[wname].items():
                row[f"{wname}_{k}"] = v
        score_rows.append(row)
        print(
            f"  {cell['name']}: n={row['n_raw']} econ_dual={both_econ} pulse_dual={both_pulse} "
            f"MJ mean={row.get('may_jul09_mean')} day_win={row.get('may_jul09_day_win')} "
            f"J10 n={row.get('jul10_23_n')} mean={row.get('jul10_23_mean')}",
            flush=True,
        )
        if both_econ:
            dual_pass.append(row)
            trade_dumps[cell["name"]] = pd.DataFrame(sized_all)

    sb = pd.DataFrame(score_rows)
    sb.to_csv(out / "scoreboard.csv", index=False)
    for name, df in trade_dumps.items():
        df.to_csv(out / f"trades_{name}.csv", index=False)

    freeze = [r for r in score_rows if r.get("freeze")]
    freeze_pass = [r for r in freeze if r.get("econ_dual_pass")]
    pulse_pass_names = [r["name"] for r in score_rows if r.get("pulse_dual_pass")]
    (out / "dual_pass.json").write_text(
        json.dumps(dual_pass, indent=2, default=str), encoding="utf-8"
    )
    summary = {
        "protocol": "am_pocket_trades_dual_offline_gate",
        "mark": "option_trade_last_slip",
        "slip": float(args.slip),
        "portfolio": {
            "position_frac": float(args.position_frac),
            "max_concurrent": int(args.max_concurrent),
            "cooldown_minutes": float(args.cooldown_minutes),
        },
        "windows": [list(w) for w in WINDOWS],
        "n_pocket_probes": len(probes),
        "n_with_path": len(base_rows),
        "coverage_note": "trade prints cover open; quotes do not",
        "freeze_cell": freeze[0]["name"] if freeze else None,
        "freeze_econ_dual_pass": bool(freeze_pass),
        "freeze_pulse_dual_pass": bool(freeze and freeze[0].get("pulse_dual_pass")),
        "freeze_blind_n": (freeze[0].get("jul10_23_n") if freeze else None),
        "econ_dual_pass_names": [r["name"] for r in dual_pass],
        "pulse_dual_pass_names": pulse_pass_names,
        "quote_note": (
            "09:30–10:00 historical quotes sparse; offline accept = trade-last. "
            "Live shadow may use IB NBBO when wired."
        ),
        "verdict": "PASS" if freeze_pass else "FAIL",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out}", flush=True)
    return 0 if freeze_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
