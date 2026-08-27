#!/usr/bin/env python3
"""AM pocket exits: **trade prints only** + regime / path dynamic ladders.

No quote FillSpec. Offline mark = option trade-last (+slip), same as
``scan_am_pocket_trades_dual``.

Thesis (user pushback):
  - stop blocking research on 09:30 quotes
  - do not freeze one TP/SL for all tapes — pick ladder by entry regime,
    optionally adapt in the first ~45s of the option path

Regimes (causal @ entry from enriched stock features):
  CHOP     — weak volr / small FO → tight quick scalp
  TREND    — fo+volr medium → scale-out + trail runner
  IMPULSE  — hot volr+FO+accel → smaller first take, wide runner

Path adapt (optional):
  t≤45s option ret ≤ −5% → fail-fast flatten
  t≤45s option ret ≥ +4% and regime≠IMPULSE → upgrade runner (wider)

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_pocket_regime_ladder \\
    --tag research_am_pocket_regime_ladder
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
from maga7.tools.scan_am_pocket_exit_design import _path_window
from maga7.tools.scan_am_pocket_multi_gate import build_gates
from maga7.tools.scan_am_pocket_risk_optimize import POCKET_SETS, _equity_stats, _month_compounds
from maga7.tools.scan_am_pocket_scaleout import simulate_scaleout
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

# Ladder specs per regime (all on trade-last path)
LADDERS: dict[str, dict[str, Any]] = {
    "CHOP": {
        "mode": "tpsl",
        "tp": 0.06,
        "sl": 0.10,
        "max_hold": 180,
    },
    "TREND": {
        "mode": "scale",
        "frac1": 0.67,
        "tp1": 0.08,
        "sl": 0.15,
        "max_hold": 480,
        "runner": "trail",
        "arm": 0.15,
        "trail": 0.12,
        "tp2": 0.25,
        "be_after_scale": True,
        "floor": 0.0,
    },
    "IMPULSE": {
        "mode": "scale",
        "frac1": 0.40,
        "tp1": 0.10,
        "sl": 0.18,
        "max_hold": 600,
        "runner": "trail",
        "arm": 0.20,
        "trail": 0.15,
        "tp2": 0.35,
        "be_after_scale": True,
        "floor": 0.0,
    },
}


def classify_regime(row: pd.Series) -> str:
    fo = abs(float(row.get("fo_vwap30") or 0.0))
    volr = float(row.get("volume_ratio_60") or 0.0)
    accel = float(row.get("accel_10_30") or 0.0)
    if not np.isfinite(volr):
        volr = 0.0
    if not np.isfinite(accel):
        accel = 0.0
    if volr >= 1.5 and fo >= 0.010 and accel >= 1e-4:
        return "IMPULSE"
    if volr >= 1.2 and fo >= 0.005:
        return "TREND"
    return "CHOP"


def _run_ladder(
    rets: np.ndarray,
    holds: np.ndarray,
    *,
    regime: str,
    path_adapt: bool,
) -> dict[str, Any]:
    reg = regime
    reason_adapt = None
    if path_adapt and len(holds) > 2:
        # first print at/after 45s
        i45 = int(np.searchsorted(holds, 45.0, side="left"))
        i45 = min(max(i45, 1), len(rets) - 1)
        r45 = float(rets[i45])
        if r45 <= -0.05:
            return {
                "ret": r45,
                "reason": "adapt_failfast",
                "hold_sec": float(holds[i45]),
                "regime": reg,
                "scaled": False,
            }
        if r45 >= 0.04 and reg != "IMPULSE":
            reg = "IMPULSE"
            reason_adapt = "adapt_upgrade"

    spec = LADDERS[reg]
    if spec["mode"] == "tpsl":
        # walk tpsl on arrays
        tp, sl, mh = float(spec["tp"]), float(spec["sl"]), float(spec["max_hold"])
        peak = -1.0
        for i in range(1, len(rets)):
            r = float(rets[i])
            h = float(holds[i])
            peak = max(peak, r)
            if r >= tp:
                return {
                    "ret": r,
                    "reason": "tp" if reason_adapt is None else f"tp+{reason_adapt}",
                    "hold_sec": h,
                    "regime": reg,
                    "scaled": False,
                }
            if r <= -sl:
                return {
                    "ret": r,
                    "reason": "sl" if reason_adapt is None else f"sl+{reason_adapt}",
                    "hold_sec": h,
                    "regime": reg,
                    "scaled": False,
                }
            if h >= mh:
                return {
                    "ret": float(rets[i - 1]),
                    "reason": "max_hold",
                    "hold_sec": float(holds[i - 1]),
                    "regime": reg,
                    "scaled": False,
                }
        return {
            "ret": float(rets[-1]),
            "reason": "max_hold",
            "hold_sec": float(holds[-1]),
            "regime": reg,
            "scaled": False,
        }

    out = simulate_scaleout(
        rets,
        holds,
        frac1=float(spec["frac1"]),
        tp1=float(spec["tp1"]),
        sl=float(spec["sl"]),
        max_hold=float(spec["max_hold"]),
        runner=str(spec["runner"]),
        tp2=float(spec.get("tp2", 9.0)),
        arm=float(spec.get("arm", 0.0)),
        trail=float(spec.get("trail", 9.0)),
        floor=float(spec.get("floor", -9.0)),
        be_after_scale=bool(spec.get("be_after_scale", False)),
    )
    out["regime"] = reg
    if reason_adapt:
        out["reason"] = f"{out['reason']}+{reason_adapt}"
    return out


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
    ap.add_argument("--tag", default="research_am_pocket_regime_ladder")
    ap.add_argument("--entry", default="vd+cont60+mf100+volr12", help="champ gate name")
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--position-frac", type=float, default=0.20)
    ap.add_argument("--max-concurrent", type=int, default=5)
    ap.add_argument("--cooldown-minutes", type=float, default=10.0)
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
    gfn = dict(build_gates())[str(args.entry)]
    probes = probes[probes.apply(gfn, axis=1)].copy()
    print(f"entry={args.entry} probes={len(probes)} (trade-mark only, no quotes)", flush=True)

    tcache: dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    def paths_for(date: str, sym: str):
        key = (date, sym)
        if key not in tcache:
            tday = load_option_trades(trades_root, sym, date)
            tcache[key] = (
                _paths_by_ticker(tday) if tday is not None and not tday.empty else {}
            )
        return tcache[key]

    variants = (
        ("fixed_tp8", False, False),  # baseline
        ("regime_ladder", True, False),
        ("regime_ladder_adapt", True, True),
    )

    score_rows: list[dict[str, Any]] = []
    for vname, use_regime, path_adapt in variants:
        win_raw: dict[str, list[dict[str, Any]]] = {w[0]: [] for w in WINDOWS}
        regime_counts: dict[str, int] = {"CHOP": 0, "TREND": 0, "IMPULSE": 0}
        for _, r in probes.iterrows():
            date, sym = str(r["date"]), str(r["symbol"])
            w = _window_of(date)
            if w is None:
                continue
            ticker = str(r["ticker"]).replace("O:", "")
            arr = paths_for(date, sym).get(ticker)
            if arr is None:
                continue
            et = to_ny(pd.Timestamp(r["entry_ts"]))
            if not use_regime:
                sim = simulate_trade_tpsl(
                    arr[0], arr[1], et, tp=0.08, sl=0.15, max_hold_sec=240, slip=float(args.slip)
                )
                if sim is None or not np.isfinite(sim["ret"]):
                    continue
                reg = "FIXED"
                hold = float(sim["hold_sec"])
                ret = float(sim["ret"])
                reason = str(sim["reason"])
            else:
                pw = _path_window(arr[0], arr[1], et, max_hold_sec=600, slip=float(args.slip))
                if pw is None:
                    continue
                rets, holds, _, _ = pw
                reg0 = classify_regime(r)
                regime_counts[reg0] = regime_counts.get(reg0, 0) + 1
                sim = _run_ladder(rets, holds, regime=reg0, path_adapt=path_adapt)
                if sim is None or not np.isfinite(sim.get("ret", float("nan"))):
                    continue
                reg = str(sim.get("regime") or reg0)
                hold = float(sim["hold_sec"])
                ret = float(sim["ret"])
                reason = str(sim["reason"])

            win_raw[w].append(
                {
                    "date": date,
                    "symbol": sym,
                    "dir": str(r["dir"]),
                    "session": str(r["session"]),
                    "entry_ts": et,
                    "exit_ts": et + pd.Timedelta(seconds=hold),
                    "ticker": ticker,
                    "ret": ret,
                    "exit_reason": reason,
                    "hold_sec": hold,
                    "oracle_ret": float(r["oracle_ret"]),
                    "regime": reg,
                    "variant": vname,
                    "window": w,
                    "event_source": "am_pocket_regime_ladder",
                }
            )

        win_stats: dict[str, dict[str, Any]] = {}
        sized_all: list[dict[str, Any]] = []
        for wname, _, _ in WINDOWS:
            raw = win_raw[wname]
            by_d: dict[str, list] = {}
            for tr in raw:
                by_d.setdefault(str(tr["date"]), []).append(tr)
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

        disc_c = float(win_stats["may_jul09"].get("eq_compound") or 0)
        blind_c = float(win_stats["jul10_23"].get("eq_compound") or 0)
        n_d = int(win_stats["may_jul09"].get("n") or 0)
        n_b = int(win_stats["jul10_23"].get("n") or 0)
        both_econ = bool(n_d >= 8 and n_b >= 3 and disc_c > 0 and blind_c > 0)
        both_pulse = True
        for wname, _, _ in WINDOWS:
            mn = 8 if wname == "may_jul09" else 6
            if not _ok(win_stats[wname], min_n=mn, min_day_win=0.55):
                both_pulse = False
                break

        if sized_all:
            o = np.array([t["oracle_ret"] for t in sized_all], dtype=float)
            rr = np.array([t["ret"] for t in sized_all], dtype=float)
            mean_cap = float(rr.mean() / o.mean()) if np.nanmean(o) > 0 else float("nan")
            pd.DataFrame(sized_all).to_csv(out / f"trades_{vname}.csv", index=False)
        else:
            mean_cap = float("nan")

        months = _month_compounds(pd.DataFrame(sized_all)) if sized_all else {}
        row: dict[str, Any] = {
            "variant": vname,
            "use_regime": use_regime,
            "path_adapt": path_adapt,
            "econ_dual_pass": both_econ,
            "pulse_dual_pass": both_pulse,
            "n_raw": sum(len(v) for v in win_raw.values()),
            "mean_capture": mean_cap,
            "regime_counts": json.dumps(regime_counts),
            "may": months.get("2026-05"),
            "jun": months.get("2026-06"),
            "jul": months.get("2026-07"),
            "disc_compound": disc_c,
            "blind_compound": blind_c,
            "disc_n": n_d,
            "blind_n": n_b,
        }
        for wname, _, _ in WINDOWS:
            for k, v in win_stats[wname].items():
                row[f"{wname}_{k}"] = v
        score_rows.append(row)
        print(
            f"{vname}: n={row['n_raw']} disc={disc_c:+.3f} blind={blind_c:+.3f} "
            f"cap={mean_cap:.3f} econ={both_econ} regimes={regime_counts}",
            flush=True,
        )

    sb = pd.DataFrame(score_rows)
    sb.to_csv(out / "scoreboard.csv", index=False)
    base = sb[sb.variant == "fixed_tp8"].iloc[0] if len(sb) else None
    best = None
    for _, r in sb.iterrows():
        if r["variant"] == "fixed_tp8":
            continue
        if not r["econ_dual_pass"]:
            continue
        if base is not None and float(r["mean_capture"]) <= float(base["mean_capture"]) + 1e-6:
            # need capture lift OR compound lift without capture drop
            if float(r["disc_compound"]) <= float(base["disc_compound"]) + 1e-9:
                continue
        if best is None or float(r["mean_capture"]) > float(best["mean_capture"]):
            best = r

    summary = {
        "protocol": "am_pocket_regime_ladder_trade_only",
        "mark": "option_trade_last_slip",
        "quote": "NOT_USED",
        "entry": args.entry,
        "ladders": LADDERS,
        "scoreboard": score_rows,
        "promote": "NONE",
        "best": None if best is None else best.to_dict(),
        "note": (
            "Promote if econ dual PASS and (capture > fixed_tp8 OR "
            "disc compound > fixed with capture not worse)."
        ),
    }
    if best is not None:
        summary["promote"] = f"RESEARCH_{best['variant']}"
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({"promote": summary["promote"], "best": summary["best"]}, indent=2, default=str))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
