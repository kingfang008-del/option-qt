#!/usr/bin/env python3
"""AM pocket call↔put flip / short-hold scalp (path oscillation regime).

Prior grids assumed one-sided hold to TP8/SL15. Option marks often swing
±30% inside 30–60s; this scan tests causal flip and short-hold policies:

  - static_tp8: frozen baseline (one side, TP8/SL15/h240)
  - scalp_h30: one side, tight TP/SL, max_hold 30s
  - flip_once: primary until adverse thr → exit → opposite short scalp
  - pingpong: up to N flips between call/put on adverse or TP, window W
  - opp_first: if primary dips thr in first T sec, abandon and ride opposite

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_pocket_flip \\
    --tag research_am_pocket_flip
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


def _ret_at(rets: np.ndarray, holds: np.ndarray, t: float) -> float:
    j = int(np.searchsorted(holds, t, side="right") - 1)
    if j < 0:
        j = 0
    if j >= len(rets):
        j = len(rets) - 1
    return float(rets[j])


def _slice_from(
    rets: np.ndarray,
    holds: np.ndarray,
    t0: float,
    *,
    slip_reentry: float = 0.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Rebase path to new entry at hold≈t0 (optional extra round-trip slip)."""
    j = int(np.searchsorted(holds, t0, side="left"))
    if j >= len(holds):
        return None
    if j == 0:
        r = rets.copy()
        h = holds.copy()
    else:
        # mark at j relative to original entry; convert to new entry at j
        # original: sell_j / entry0 - 1 = rets[j]
        # new entry buy ≈ sell_j * (1+slip)/(1-slip) ≈ sell_j * (1+2slip) for small slip
        # simpler: rebase so ret'=0 at j, then scale subsequent relative moves
        base = 1.0 + float(rets[j])
        if not np.isfinite(base) or base <= 0:
            return None
        r = ((1.0 + rets[j:]) / base) - 1.0
        # pay re-entry friction once
        r = (1.0 + r) / (1.0 + float(slip_reentry)) - 1.0
        h = holds[j:] - holds[j]
    return r.astype(np.float64), h.astype(np.float64)


def simulate_scalp(
    rets: np.ndarray,
    holds: np.ndarray,
    *,
    tp: float,
    sl: float,
    max_hold: float,
) -> dict[str, Any]:
    return simulate_exit(
        rets, holds, mode="tpsl", params={"tp": tp, "sl": sl, "max_hold": max_hold}
    )


def simulate_flip_once(
    prim: tuple[np.ndarray, np.ndarray],
    opp: tuple[np.ndarray, np.ndarray],
    *,
    adv: float,
    tp: float,
    sl: float,
    max_leg: float,
    window: float,
    reentry_slip: float,
) -> dict[str, Any]:
    """Primary until -adv or tp/max_leg; on adverse flip to opposite once."""
    pr, ph = prim
    legs: list[dict[str, Any]] = []
    sim1 = simulate_scalp(pr, ph, tp=tp, sl=adv, max_hold=min(max_leg, window))
    legs.append({"side": "prim", **sim1})
    total = float(sim1["ret"])
    t_used = float(sim1["hold_sec"])
    flipped = False
    if sim1["reason"] == "sl" and t_used < window - 5:
        sliced = _slice_from(opp[0], opp[1], t_used, slip_reentry=reentry_slip)
        if sliced is not None:
            o_r, o_h = sliced
            remain = window - t_used
            sim2 = simulate_scalp(o_r, o_h, tp=tp, sl=sl, max_hold=min(max_leg, remain))
            legs.append({"side": "opp", **sim2})
            total = total + float(sim2["ret"])
            t_used = t_used + float(sim2["hold_sec"])
            flipped = True
    return {
        "ret": total,
        "hold_sec": t_used,
        "reason": "flip" if flipped else str(sim1["reason"]),
        "n_legs": len(legs),
        "flipped": flipped,
        "legs": legs,
    }


def simulate_pingpong(
    prim: tuple[np.ndarray, np.ndarray],
    opp: tuple[np.ndarray, np.ndarray],
    *,
    adv: float,
    tp: float,
    sl: float,
    max_leg: float,
    window: float,
    max_flips: int,
    reentry_slip: float,
) -> dict[str, Any]:
    """Alternate sides on adverse (or after TP) up to max_flips within window."""
    sides = [prim, opp]
    names = ["prim", "opp"]
    side_i = 0
    t_cursor = 0.0
    total = 0.0
    legs: list[dict[str, Any]] = []
    flips = 0
    while t_cursor < window - 3 and flips <= max_flips:
        path = sides[side_i]
        sliced = _slice_from(path[0], path[1], t_cursor, slip_reentry=reentry_slip if legs else 0.0)
        if sliced is None:
            break
        r, h = sliced
        remain = window - t_cursor
        # first leg uses adv as SL to encourage flip; later legs use sl
        leg_sl = adv if not legs else sl
        sim = simulate_scalp(r, h, tp=tp, sl=leg_sl, max_hold=min(max_leg, remain))
        legs.append({"side": names[side_i], **sim})
        total += float(sim["ret"])
        t_cursor += float(sim["hold_sec"])
        if sim["reason"] == "sl" and flips < max_flips and t_cursor < window - 3:
            side_i = 1 - side_i
            flips += 1
            continue
        if sim["reason"] == "tp" and flips < max_flips and t_cursor < window - 3:
            # after TP, optionally reverse for mean-reversion scalp
            side_i = 1 - side_i
            flips += 1
            continue
        break
    return {
        "ret": total,
        "hold_sec": t_cursor,
        "reason": f"pp_legs{len(legs)}_flips{flips}",
        "n_legs": len(legs),
        "flipped": flips > 0,
        "legs": legs,
    }


def simulate_opp_first(
    prim: tuple[np.ndarray, np.ndarray],
    opp: tuple[np.ndarray, np.ndarray],
    *,
    look_t: float,
    dip: float,
    tp: float,
    sl: float,
    max_hold: float,
) -> dict[str, Any]:
    """If primary dips to -dip within look_t, skip primary and scalp opposite instead."""
    pr, ph = prim
    mask = ph <= look_t + 1e-9
    early = pr[mask]
    if len(early) and float(np.nanmin(early)) <= -dip:
        return {
            **simulate_scalp(opp[0], opp[1], tp=tp, sl=sl, max_hold=max_hold),
            "flipped": True,
            "n_legs": 1,
            "reason_tag": "opp_first",
        }
    sim = simulate_scalp(pr, ph, tp=tp, sl=sl, max_hold=max_hold)
    return {**sim, "flipped": False, "n_legs": 1, "reason_tag": "prim"}


def _policy_grid() -> list[dict[str, Any]]:
    cfgs: list[dict[str, Any]] = [
        {"name": "static_tp8_sl15_h240", "mode": "static", "tp": 0.08, "sl": 0.15, "max_hold": 240},
        {"name": "scalp_tp8_sl10_h30", "mode": "static", "tp": 0.08, "sl": 0.10, "max_hold": 30},
        {"name": "scalp_tp10_sl8_h30", "mode": "static", "tp": 0.10, "sl": 0.08, "max_hold": 30},
        {"name": "scalp_tp6_sl6_h30", "mode": "static", "tp": 0.06, "sl": 0.06, "max_hold": 30},
        {"name": "scalp_tp8_sl10_h45", "mode": "static", "tp": 0.08, "sl": 0.10, "max_hold": 45},
        {"name": "scalp_tp12_sl10_h30", "mode": "static", "tp": 0.12, "sl": 0.10, "max_hold": 30},
    ]
    for adv in (0.06, 0.08, 0.10, 0.12):
        for tp in (0.08, 0.10, 0.12, 0.15):
            for max_leg in (20, 30, 45):
                for window in (60, 90, 120):
                    cfgs.append(
                        {
                            "name": f"flip1_a{adv:g}_tp{tp:g}_leg{max_leg}_w{window}",
                            "mode": "flip_once",
                            "adv": adv,
                            "tp": tp,
                            "sl": 0.10,
                            "max_leg": max_leg,
                            "window": window,
                        }
                    )
    for adv, tp, max_leg, window, nflip in (
        (0.08, 0.08, 30, 90, 2),
        (0.08, 0.10, 30, 90, 2),
        (0.08, 0.10, 30, 120, 3),
        (0.10, 0.10, 30, 90, 2),
        (0.06, 0.08, 20, 60, 2),
        (0.08, 0.12, 30, 120, 2),
        (0.10, 0.15, 45, 120, 2),
    ):
        cfgs.append(
            {
                "name": f"pp_a{adv:g}_tp{tp:g}_leg{max_leg}_w{window}_n{nflip}",
                "mode": "pingpong",
                "adv": adv,
                "tp": tp,
                "sl": 0.10,
                "max_leg": max_leg,
                "window": window,
                "max_flips": nflip,
            }
        )
    for look_t, dip in ((10, 0.08), (15, 0.08), (15, 0.10), (20, 0.10)):
        for tp, sl, mh in ((0.10, 0.10, 30), (0.12, 0.10, 45), (0.08, 0.10, 30)):
            cfgs.append(
                {
                    "name": f"opp1st_t{look_t}_d{dip:g}_tp{tp:g}_sl{sl:g}_h{mh}",
                    "mode": "opp_first",
                    "look_t": look_t,
                    "dip": dip,
                    "tp": tp,
                    "sl": sl,
                    "max_hold": mh,
                }
            )
    return cfgs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--enriched", default=str(DEFAULT_ENRICHED))
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--tag", default="research_am_pocket_flip")
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
    ap.add_argument("--reentry-slip", type=float, default=0.02, help="extra friction on flip re-entry")
    ap.add_argument("--entry", default=CHAMP, help="gate name or vd_soft")
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    out = Path(prof["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    trades_root = Path(args.trades_root)
    lock = load_multidte_lock_index(Path(prof["_paths"]["open_locked_map"]).expanduser())

    probes = pd.read_csv(args.enriched)
    probes["entry_ts"] = pd.to_datetime(probes["entry_ts"])
    probes = probes[probes["enrich_ok"] == True].copy()  # noqa: E712
    pdf = pd.DataFrame(sorted(POCKET_SETS["no_b_up"]), columns=["session", "tod_bucket", "dir"])
    probes = probes.merge(pdf, on=["session", "tod_bucket", "dir"], how="inner")
    gate = dict(build_gates())[str(args.entry)]
    probes = probes[probes.apply(gate, axis=1)].copy()
    print(f"entry={args.entry} probes={len(probes)}", flush=True)

    path_cache: dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    def paths_for(date: str, sym: str):
        key = (date, sym)
        if key not in path_cache:
            tday = load_option_trades(trades_root, sym, date)
            path_cache[key] = (
                _paths_by_ticker(tday) if tday is not None and not tday.empty else {}
            )
        return path_cache[key]

    prepared: list[dict[str, Any]] = []
    for _, r in probes.iterrows():
        date, sym = str(r["date"]), str(r["symbol"])
        et = to_ny(pd.Timestamp(r["entry_ts"]))
        direction = str(r["dir"])
        opp_dir = "DN" if direction == "UP" else "UP"
        spot = float(r["spot"]) if np.isfinite(float(r.get("spot") or np.nan)) else None
        by_dte = lock.get((sym, date))
        arrs = paths_for(date, sym)
        prim_key = str(r["ticker"]).replace("O:", "")
        opp_t, _, _ = resolve_open_lock_contract(
            by_dte,
            direction=opp_dir,
            moneyness="ATM",
            spot=spot,
            prefer_dte=0,
            allowed_dte=(0, 1, 2),
            clear_otm_thresh=0.01,
            ladder=True,
            otm_rungs=3,
        )
        if not opp_t:
            continue
        p_path = arrs.get(prim_key)
        o_path = arrs.get(str(opp_t).replace("O:", ""))
        if p_path is None or o_path is None:
            continue
        pwin = _path_window(p_path[0], p_path[1], et, max_hold_sec=180, slip=float(args.slip))
        owin = _path_window(o_path[0], o_path[1], et, max_hold_sec=180, slip=float(args.slip))
        if pwin is None or owin is None:
            continue
        prepared.append(
            {
                "row": r,
                "date": date,
                "symbol": sym,
                "dir": direction,
                "session": str(r["session"]),
                "calendar": str(r["calendar"]),
                "entry_ts": et,
                "prim": (pwin[0], pwin[1]),
                "opp": (owin[0], owin[1]),
                "oracle_ret": float(r["oracle_ret"]),
            }
        )
    print(f"prepared dual-paths={len(prepared)}", flush=True)

    policies = _policy_grid()
    print(f"policies={len(policies)}", flush=True)
    score_rows: list[dict[str, Any]] = []

    for pol in policies:
        raw: list[dict[str, Any]] = []
        n_flip = 0
        n_legs = 0
        for p in prepared:
            if pol["mode"] == "static":
                sim = simulate_scalp(
                    p["prim"][0],
                    p["prim"][1],
                    tp=float(pol["tp"]),
                    sl=float(pol["sl"]),
                    max_hold=float(pol["max_hold"]),
                )
                sim = {**sim, "flipped": False, "n_legs": 1}
            elif pol["mode"] == "flip_once":
                sim = simulate_flip_once(
                    p["prim"],
                    p["opp"],
                    adv=float(pol["adv"]),
                    tp=float(pol["tp"]),
                    sl=float(pol["sl"]),
                    max_leg=float(pol["max_leg"]),
                    window=float(pol["window"]),
                    reentry_slip=float(args.reentry_slip),
                )
            elif pol["mode"] == "pingpong":
                sim = simulate_pingpong(
                    p["prim"],
                    p["opp"],
                    adv=float(pol["adv"]),
                    tp=float(pol["tp"]),
                    sl=float(pol["sl"]),
                    max_leg=float(pol["max_leg"]),
                    window=float(pol["window"]),
                    max_flips=int(pol["max_flips"]),
                    reentry_slip=float(args.reentry_slip),
                )
            elif pol["mode"] == "opp_first":
                sim = simulate_opp_first(
                    p["prim"],
                    p["opp"],
                    look_t=float(pol["look_t"]),
                    dip=float(pol["dip"]),
                    tp=float(pol["tp"]),
                    sl=float(pol["sl"]),
                    max_hold=float(pol["max_hold"]),
                )
            else:
                continue
            if not np.isfinite(sim.get("ret", np.nan)):
                continue
            if sim.get("flipped"):
                n_flip += 1
            n_legs += int(sim.get("n_legs") or 1)
            et = p["entry_ts"]
            raw.append(
                {
                    "date": p["date"],
                    "symbol": p["symbol"],
                    "dir": p["dir"],
                    "session": p["session"],
                    "calendar": p["calendar"],
                    "entry_ts": et,
                    "exit_ts": et + pd.Timedelta(seconds=float(sim["hold_sec"])),
                    "ret": float(sim["ret"]),
                    "exit_reason": str(sim.get("reason") or sim.get("reason_tag") or ""),
                    "hold_sec": float(sim["hold_sec"]),
                    "oracle_ret": float(p["oracle_ret"]),
                    "flipped": bool(sim.get("flipped")),
                    "n_legs": int(sim.get("n_legs") or 1),
                }
            )

        disc = [t for t in raw if t["calendar"] == "may_jul09"]
        blind = [t for t in raw if t["calendar"] == "jul10_23"]
        sized_d = _portfolio_day(
            sorted(disc, key=lambda x: (x["entry_ts"], x["symbol"])),
            position_frac=float(args.position_frac),
            max_concurrent=int(args.max_concurrent),
            cooldown_minutes=5.0,
        )
        sized_b = _portfolio_day(
            sorted(blind, key=lambda x: (x["entry_ts"], x["symbol"])),
            position_frac=float(args.position_frac),
            max_concurrent=int(args.max_concurrent),
            cooldown_minutes=5.0,
        )
        st_d = _equity_stats(pd.DataFrame(sized_d))
        st_b = _equity_stats(pd.DataFrame(sized_b))
        months = _month_compounds(pd.DataFrame(sized_d + sized_b))
        if raw:
            o = np.array([t["oracle_ret"] for t in raw], dtype=float)
            rr = np.array([t["ret"] for t in raw], dtype=float)
            mean_cap = float(rr.mean() / o.mean()) if o.mean() > 0 else float("nan")
            mean_hold = float(np.mean([t["hold_sec"] for t in raw]))
            mean_legs = float(np.mean([t["n_legs"] for t in raw]))
        else:
            mean_cap = mean_hold = mean_legs = float("nan")
        row: dict[str, Any] = {
            "policy": pol["name"],
            "mode": pol["mode"],
            "n_raw": len(raw),
            "frac_flipped": (n_flip / len(raw)) if raw else 0.0,
            "mean_legs": mean_legs,
            "mean_hold": mean_hold,
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

    sb = pd.DataFrame(score_rows)
    sb.to_csv(out / "scoreboard.csv", index=False)

    base = sb[sb.policy == "static_tp8_sl15_h240"]
    base_row = base.iloc[0].to_dict() if len(base) else {}
    bw = float(base_row.get("disc_trade_win") or 0)
    bdd = float(base_row.get("disc_maxdd") or -1)
    bcmp = float(base_row.get("disc_compound") or 0)
    bcap = float(base_row.get("mean_capture") or 0)

    soft = sb[
        (sb["disc_n"].fillna(0) >= 15)
        & (sb["disc_trade_win"].fillna(0) >= 0.60)
        & (sb["disc_maxdd"].fillna(-1) >= -0.22)
        & (sb["disc_compound"].fillna(0) > 0)
        & (sb["may"].fillna(0) > 0)
    ].copy()
    if not soft.empty:
        soft["score"] = (
            soft["disc_trade_win"] * 0.15
            + (1 + soft["disc_maxdd"]) * 0.2
            + np.clip(soft["disc_compound"], 0, 2.0) / 2.0 * 0.25
            + np.clip(soft["mean_capture"], 0, 0.5) / 0.5 * 0.4
        )
        soft = soft.sort_values("score", ascending=False)

    better = sb[
        (sb["disc_n"].fillna(0) >= 15)
        & (sb["disc_trade_win"].fillna(0) >= bw - 0.08)
        & (sb["disc_maxdd"].fillna(-1) >= -0.22)
        & (
            (sb["mean_capture"].fillna(0) > bcap + 0.03)
            | (sb["disc_compound"].fillna(0) > bcmp + 0.08)
        )
        & (sb["disc_compound"].fillna(0) > 0)
    ].sort_values(["mean_capture", "disc_compound"], ascending=[False, False])

    # best by mode
    by_mode = {}
    for mode, g in sb.groupby("mode"):
        g2 = g[g["disc_compound"].fillna(-9) > -9].sort_values("disc_compound", ascending=False)
        by_mode[str(mode)] = g2.head(5).to_dict(orient="records") if len(g2) else []

    verdict = {
        "protocol": "call_put_flip_short_hold_on_champ",
        "entry": args.entry,
        "baseline": base_row,
        "top_soft": soft.head(15).to_dict(orient="records") if len(soft) else [],
        "better_than_static": better.head(15).to_dict(orient="records") if len(better) else [],
        "best_by_mode": by_mode,
        "n_policies": len(policies),
        "note": "rets are summed leg returns (not compounded within episode)",
    }
    (out / "summary.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")

    cols = [
        c
        for c in [
            "policy",
            "mode",
            "n_raw",
            "frac_flipped",
            "mean_legs",
            "mean_hold",
            "disc_trade_win",
            "disc_maxdd",
            "disc_compound",
            "mean_capture",
            "blind_compound",
            "may",
            "jun",
            "jul",
        ]
        if c in sb.columns
    ]
    print("\nBASELINE", flush=True)
    print(base[cols].to_string(index=False) if len(base) else "(none)", flush=True)
    print("\nTOP soft", flush=True)
    print(soft[cols].head(12).to_string(index=False) if len(soft) else "(none)", flush=True)
    print("\nBETTER capture/compound", flush=True)
    print(better[cols].head(12).to_string(index=False) if len(better) else "(none)", flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
