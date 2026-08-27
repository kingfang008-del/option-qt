#!/usr/bin/env python3
"""Quote dual validation for AM pocket opp_first / flip-on-dip.

Trade-last ``opp_first`` peeked the next look_t seconds before choosing side
(non-causal). This tool validates executable quote FillSpec paths:

  - static: one-sided quote TP/SL
  - flip_causal: enter primary; if mark hits -dip within look_t, exit and
    enter opposite for a short scalp (eats the adverse leg)
  - opp_opt (diagnostic only): skip primary if min(first look_t)<=-dip
    — same lookahead as trade-last; NOT live-feasible

Dual windows: may_jul09 / jul10_23. PASS via delayed-confirm quote dual _ok.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_pocket_flip_quote_dual \\
    --tag research_am_pocket_flip_quote_dual
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
from maga7.common.fills import FillSpec
from maga7.common.open_lock import load_multidte_lock_index, resolve_open_lock_contract
from maga7.common.option_quote_tpsl import entry_quote_row, simulate_quote_tpsl
from maga7.common.replay import load_quotes, path_for_ticker, to_ny
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_delayed_confirm_quote_dual import _ok, _prep_path, _stats
from maga7.tools.scan_am_pocket_multi_gate import build_gates
from maga7.tools.scan_am_pocket_risk_optimize import POCKET_SETS, _equity_stats, _month_compounds

DEFAULT_ENRICHED = Path(
    "/mnt/s990/data/maga7/results/research_am_pocket_multi_gate/enriched_probes.csv"
)
PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
WINDOWS = (
    ("may_jul09", "2026-05-01", "2026-07-09"),
    ("jul10_23", "2026-07-10", "2026-07-23"),
)
NY = "America/New_York"


def _window_of(date: str) -> str | None:
    for name, a, b in WINDOWS:
        if a <= date <= b:
            return name
    return None


def _policies() -> list[dict[str, Any]]:
    return [
        {
            "name": "static_tp8_sl15_h240",
            "mode": "static",
            "tp": 0.08,
            "sl": 0.15,
            "max_hold": 240,
            "causal": True,
        },
        {
            "name": "static_tp12_sl10_h45",
            "mode": "static",
            "tp": 0.12,
            "sl": 0.10,
            "max_hold": 45,
            "causal": True,
        },
        # causal flip-on-dip (live-feasible shape)
        {
            "name": "flip_t10_d08_opp_tp12_sl10_h45",
            "mode": "flip_causal",
            "look_t": 10,
            "dip": 0.08,
            "prim_tp": 0.08,
            "prim_sl": 0.15,
            "prim_max_hold": 240,
            "opp_tp": 0.12,
            "opp_sl": 0.10,
            "opp_max_hold": 45,
            "causal": True,
        },
        {
            "name": "flip_t10_d08_opp_tp10_sl10_h30",
            "mode": "flip_causal",
            "look_t": 10,
            "dip": 0.08,
            "prim_tp": 0.08,
            "prim_sl": 0.15,
            "prim_max_hold": 240,
            "opp_tp": 0.10,
            "opp_sl": 0.10,
            "opp_max_hold": 30,
            "causal": True,
        },
        {
            "name": "flip_t15_d08_opp_tp12_sl10_h45",
            "mode": "flip_causal",
            "look_t": 15,
            "dip": 0.08,
            "prim_tp": 0.08,
            "prim_sl": 0.15,
            "prim_max_hold": 240,
            "opp_tp": 0.12,
            "opp_sl": 0.10,
            "opp_max_hold": 45,
            "causal": True,
        },
        {
            "name": "flip_t10_d10_opp_tp12_sl10_h45",
            "mode": "flip_causal",
            "look_t": 10,
            "dip": 0.10,
            "prim_tp": 0.08,
            "prim_sl": 0.15,
            "prim_max_hold": 240,
            "opp_tp": 0.12,
            "opp_sl": 0.10,
            "opp_max_hold": 45,
            "causal": True,
        },
        # during look window use dip as hard SL then flip; after look use prim tpsl
        {
            "name": "flip_t10_d08_both_short",
            "mode": "flip_causal",
            "look_t": 10,
            "dip": 0.08,
            "prim_tp": 0.12,
            "prim_sl": 0.10,
            "prim_max_hold": 45,
            "opp_tp": 0.12,
            "opp_sl": 0.10,
            "opp_max_hold": 45,
            "causal": True,
        },
        # optimistic lookahead (NOT causal) — trade-last parity diagnostic
        {
            "name": "opp_opt_t10_d08_tp12_sl10_h45",
            "mode": "opp_opt",
            "look_t": 10,
            "dip": 0.08,
            "tp": 0.12,
            "sl": 0.10,
            "max_hold": 45,
            "prim_tp": 0.08,
            "prim_sl": 0.15,
            "prim_max_hold": 240,
            "causal": False,
        },
    ]


def simulate_quote_flip_causal(
    prim_path: pd.DataFrame,
    opp_path: pd.DataFrame,
    entry_ts: pd.Timestamp,
    *,
    look_t: float,
    dip: float,
    prim_tp: float,
    prim_sl: float,
    prim_max_hold: int,
    opp_tp: float,
    opp_sl: float,
    opp_max_hold: int,
    fill: FillSpec,
    max_lag_sec: float,
    max_spread_pct: float,
    min_mid: float,
) -> dict[str, Any] | None:
    """Enter primary; on -dip within look_t, exit and scalp opposite."""
    ent = entry_quote_row(
        prim_path,
        entry_ts,
        max_lag_sec=max_lag_sec,
        max_spread_pct=max_spread_pct,
        min_mid=min_mid,
    )
    if ent is None:
        return None
    after: pd.DataFrame = ent["after"]
    entry_px = fill.buy(ent["bid"], ent["ask"])
    if not np.isfinite(entry_px) or entry_px <= 0:
        return None
    t_entry = ent["entry_ts"]
    look_end = t_entry + pd.Timedelta(seconds=float(look_t))
    hard_end = t_entry + pd.Timedelta(seconds=int(prim_max_hold))

    flipped = False
    flip_ts: pd.Timestamp | None = None
    leg1_ret = float("nan")
    reason = "max_hold"
    exit_ts = t_entry
    hold = 0.0
    mfe, mae = -1.0, 1.0

    for i in range(1, len(after)):
        r = after.iloc[i]
        ts = to_ny(r["timestamp"])
        if ts > hard_end:
            prev = after.iloc[i - 1]
            px = fill.sell(float(prev["bid"]), float(prev["ask"]))
            leg1_ret = px / entry_px - 1.0
            hold = (to_ny(prev["timestamp"]) - t_entry).total_seconds()
            exit_ts = to_ny(prev["timestamp"])
            reason = "max_hold"
            break
        bid, ask = float(r["bid"]), float(r["ask"])
        if not (np.isfinite(bid) and np.isfinite(ask) and ask > bid > 0):
            continue
        px = fill.sell(bid, ask)
        cur = px / entry_px - 1.0
        mfe = max(mfe, cur)
        mae = min(mae, cur)
        # TP always honored
        if cur >= float(prim_tp):
            leg1_ret, hold, reason, exit_ts = cur, (ts - t_entry).total_seconds(), "tp", ts
            break
        # within look window: dip triggers flip (not full prim_sl yet)
        if ts <= look_end and cur <= -float(dip):
            leg1_ret, hold, reason, exit_ts = cur, (ts - t_entry).total_seconds(), "flip_dip", ts
            flipped = True
            flip_ts = ts
            break
        # after look window: normal SL
        if ts > look_end and cur <= -float(prim_sl):
            leg1_ret, hold, reason, exit_ts = cur, (ts - t_entry).total_seconds(), "sl", ts
            break
    else:
        last = after.iloc[-1]
        px = fill.sell(float(last["bid"]), float(last["ask"]))
        leg1_ret = px / entry_px - 1.0
        hold = (to_ny(last["timestamp"]) - t_entry).total_seconds()
        exit_ts = to_ny(last["timestamp"])
        reason = "max_hold"

    if not np.isfinite(leg1_ret):
        return None

    total = float(leg1_ret)
    n_legs = 1
    opp_ret = None
    opp_reason = None
    if flipped and flip_ts is not None:
        sim2 = simulate_quote_tpsl(
            opp_path,
            flip_ts,
            tp=float(opp_tp),
            sl=float(opp_sl),
            max_hold_sec=int(opp_max_hold),
            fill=fill,
            max_lag_sec=max_lag_sec,
            max_spread_pct=max_spread_pct,
            min_mid=min_mid,
        )
        if sim2 is not None:
            opp_ret = float(sim2["ret"])
            opp_reason = str(sim2["reason"])
            # sequential full-notional compound within episode
            total = (1.0 + float(leg1_ret)) * (1.0 + opp_ret) - 1.0
            hold = hold + float(sim2["hold_sec"])
            exit_ts = sim2["exit_ts"]
            n_legs = 2
            reason = f"flip+{opp_reason}"
        else:
            reason = "flip_opp_reject"

    return {
        "ret": float(total),
        "reason": reason,
        "hold_sec": float(hold),
        "mfe": float(mfe if np.isfinite(mfe) else leg1_ret),
        "mae": float(mae if np.isfinite(mae) else leg1_ret),
        "entry_lag_sec": float(ent["lag_sec"]),
        "entry_spread_pct": float(ent["spread_pct"]),
        "entry_ts": t_entry,
        "exit_ts": exit_ts,
        "flipped": flipped,
        "n_legs": n_legs,
        "leg1_ret": float(leg1_ret),
        "opp_ret": opp_ret,
        "causal": True,
    }


def simulate_quote_opp_opt(
    prim_path: pd.DataFrame,
    opp_path: pd.DataFrame,
    entry_ts: pd.Timestamp,
    *,
    look_t: float,
    dip: float,
    tp: float,
    sl: float,
    max_hold: int,
    prim_tp: float,
    prim_sl: float,
    prim_max_hold: int,
    fill: FillSpec,
    max_lag_sec: float,
    max_spread_pct: float,
    min_mid: float,
) -> dict[str, Any] | None:
    """Non-causal: if primary would dip within look_t, trade opposite from t0."""
    # Probe primary marks without committing — uses same entry quote then scan
    ent = entry_quote_row(
        prim_path,
        entry_ts,
        max_lag_sec=max_lag_sec,
        max_spread_pct=max_spread_pct,
        min_mid=min_mid,
    )
    if ent is None:
        return None
    after: pd.DataFrame = ent["after"]
    entry_px = fill.buy(ent["bid"], ent["ask"])
    t_entry = ent["entry_ts"]
    look_end = t_entry + pd.Timedelta(seconds=float(look_t))
    would_dip = False
    for i in range(1, len(after)):
        r = after.iloc[i]
        ts = to_ny(r["timestamp"])
        if ts > look_end:
            break
        bid, ask = float(r["bid"]), float(r["ask"])
        if not (np.isfinite(bid) and np.isfinite(ask) and ask > bid > 0):
            continue
        cur = fill.sell(bid, ask) / entry_px - 1.0
        if cur <= -float(dip):
            would_dip = True
            break
    if would_dip:
        sim = simulate_quote_tpsl(
            opp_path,
            entry_ts,
            tp=tp,
            sl=sl,
            max_hold_sec=max_hold,
            fill=fill,
            max_lag_sec=max_lag_sec,
            max_spread_pct=max_spread_pct,
            min_mid=min_mid,
        )
        if sim is None:
            return None
        return {**sim, "flipped": True, "n_legs": 1, "causal": False, "leg1_ret": None}
    sim = simulate_quote_tpsl(
        prim_path,
        entry_ts,
        tp=prim_tp,
        sl=prim_sl,
        max_hold_sec=prim_max_hold,
        fill=fill,
        max_lag_sec=max_lag_sec,
        max_spread_pct=max_spread_pct,
        min_mid=min_mid,
    )
    if sim is None:
        return None
    return {**sim, "flipped": False, "n_legs": 1, "causal": False, "leg1_ret": float(sim["ret"])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--enriched", default=str(DEFAULT_ENRICHED))
    ap.add_argument("--tag", default="research_am_pocket_flip_quote_dual")
    ap.add_argument("--entry", default="vd+cont60+mf100+volr12")
    ap.add_argument("--max-spreads", default="0.10,0.15")
    ap.add_argument("--max-lags", default="3,5")
    ap.add_argument("--min-mid", type=float, default=0.05)
    ap.add_argument("--entry-frac", type=float, default=0.75)
    ap.add_argument("--exit-frac", type=float, default=0.75)
    ap.add_argument("--position-frac", type=float, default=0.20)
    ap.add_argument("--max-concurrent", type=int, default=5)
    ap.add_argument("--cooldown-minutes", type=float, default=5.0)
    ap.add_argument("--min-n", type=int, default=8)
    ap.add_argument("--min-day-win", type=float, default=0.55)
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    out = Path(prof["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    quote_root = Path(prof["_paths"]["quote_1s_root"])
    lock = load_multidte_lock_index(Path(prof["_paths"]["open_locked_map"]).expanduser())
    fill = FillSpec(entry_frac=float(args.entry_frac), exit_frac=float(args.exit_frac))
    spreads = [float(x) for x in args.max_spreads.split(",") if x.strip()]
    lags = [float(x) for x in args.max_lags.split(",") if x.strip()]

    probes = pd.read_csv(args.enriched)
    probes["entry_ts"] = pd.to_datetime(probes["entry_ts"])
    probes = probes[probes["enrich_ok"] == True].copy()  # noqa: E712
    pdf = pd.DataFrame(sorted(POCKET_SETS["no_b_up"]), columns=["session", "tod_bucket", "dir"])
    probes = probes.merge(pdf, on=["session", "tod_bucket", "dir"], how="inner")
    gate = dict(build_gates())[str(args.entry)]
    probes = probes[probes.apply(gate, axis=1)].copy()
    print(f"entry={args.entry} probes={len(probes)}", flush=True)

    # resolve opposite tickers + cache quote paths
    prepared: list[dict[str, Any]] = []
    qday_cache: dict[tuple[str, str], pd.DataFrame | None] = {}

    def qday(date: str, sym: str) -> pd.DataFrame | None:
        key = (date, sym)
        if key not in qday_cache:
            qday_cache[key] = _prep_path(load_quotes(quote_root, sym, date))
        return qday_cache[key]

    for _, r in probes.iterrows():
        date, sym = str(r["date"]), str(r["symbol"])
        if _window_of(date) is None:
            continue
        et = to_ny(pd.Timestamp(r["entry_ts"]))
        direction = str(r["dir"])
        opp_dir = "DN" if direction == "UP" else "UP"
        spot = float(r["spot"]) if np.isfinite(float(r.get("spot") or np.nan)) else None
        by_dte = lock.get((sym, date))
        prim_t = str(r["ticker"]).replace("O:", "")
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
        day_q = qday(date, sym)
        if day_q is None:
            continue
        pp = _prep_path(path_for_ticker(day_q, prim_t))
        op = _prep_path(path_for_ticker(day_q, str(opp_t).replace("O:", "")))
        if pp is None or op is None:
            continue
        prepared.append(
            {
                "date": date,
                "symbol": sym,
                "dir": direction,
                "session": str(r["session"]),
                "calendar": str(r["calendar"]),
                "entry_ts": et,
                "prim_path": pp,
                "opp_path": op,
                "window": _window_of(date),
            }
        )
    print(f"prepared quote dual={len(prepared)}", flush=True)

    policies = _policies()
    score_rows: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []

    for lag in lags:
        for sp in spreads:
            for pol in policies:
                raw: list[dict[str, Any]] = []
                n_rej = 0
                for p in prepared:
                    if pol["mode"] == "static":
                        sim = simulate_quote_tpsl(
                            p["prim_path"],
                            p["entry_ts"],
                            tp=float(pol["tp"]),
                            sl=float(pol["sl"]),
                            max_hold_sec=int(pol["max_hold"]),
                            fill=fill,
                            max_lag_sec=float(lag),
                            max_spread_pct=float(sp),
                            min_mid=float(args.min_mid),
                        )
                        if sim is None:
                            n_rej += 1
                            continue
                        sim = {**sim, "flipped": False, "n_legs": 1, "causal": True}
                    elif pol["mode"] == "flip_causal":
                        sim = simulate_quote_flip_causal(
                            p["prim_path"],
                            p["opp_path"],
                            p["entry_ts"],
                            look_t=float(pol["look_t"]),
                            dip=float(pol["dip"]),
                            prim_tp=float(pol["prim_tp"]),
                            prim_sl=float(pol["prim_sl"]),
                            prim_max_hold=int(pol["prim_max_hold"]),
                            opp_tp=float(pol["opp_tp"]),
                            opp_sl=float(pol["opp_sl"]),
                            opp_max_hold=int(pol["opp_max_hold"]),
                            fill=fill,
                            max_lag_sec=float(lag),
                            max_spread_pct=float(sp),
                            min_mid=float(args.min_mid),
                        )
                        if sim is None:
                            n_rej += 1
                            continue
                    elif pol["mode"] == "opp_opt":
                        sim = simulate_quote_opp_opt(
                            p["prim_path"],
                            p["opp_path"],
                            p["entry_ts"],
                            look_t=float(pol["look_t"]),
                            dip=float(pol["dip"]),
                            tp=float(pol["tp"]),
                            sl=float(pol["sl"]),
                            max_hold=int(pol["max_hold"]),
                            prim_tp=float(pol["prim_tp"]),
                            prim_sl=float(pol["prim_sl"]),
                            prim_max_hold=int(pol["prim_max_hold"]),
                            fill=fill,
                            max_lag_sec=float(lag),
                            max_spread_pct=float(sp),
                            min_mid=float(args.min_mid),
                        )
                        if sim is None:
                            n_rej += 1
                            continue
                    else:
                        continue
                    et = sim.get("entry_ts") or p["entry_ts"]
                    xt = sim.get("exit_ts") or et
                    rec = {
                        "date": p["date"],
                        "symbol": p["symbol"],
                        "dir": p["dir"],
                        "session": p["session"],
                        "calendar": p["calendar"],
                        "window": p["window"],
                        "entry_ts": et,
                        "exit_ts": xt,
                        "ret": float(sim["ret"]),
                        "exit_reason": str(sim["reason"]),
                        "hold_sec": float(sim["hold_sec"]),
                        "flipped": bool(sim.get("flipped")),
                        "n_legs": int(sim.get("n_legs") or 1),
                        "policy": pol["name"],
                        "lag": lag,
                        "spread": sp,
                        "causal": bool(pol.get("causal", True)),
                    }
                    raw.append(rec)
                    all_trades.append(rec)

                by_w: dict[str, list] = {w[0]: [] for w in WINDOWS}
                for t in raw:
                    by_w.setdefault(str(t["window"]), []).append(t)
                row: dict[str, Any] = {
                    "policy": pol["name"],
                    "mode": pol["mode"],
                    "causal": bool(pol.get("causal", True)),
                    "lag": lag,
                    "spread": sp,
                    "n_raw": len(raw),
                    "n_reject": n_rej,
                    "frac_flipped": float(np.mean([t["flipped"] for t in raw])) if raw else 0.0,
                }
                dual_ok = True
                for wname, _, _ in WINDOWS:
                    bucket = by_w.get(wname) or []
                    sized = _portfolio_day(
                        sorted(bucket, key=lambda x: (x["entry_ts"], x["symbol"])),
                        position_frac=float(args.position_frac),
                        max_concurrent=int(args.max_concurrent),
                        cooldown_minutes=float(args.cooldown_minutes),
                    )
                    st = _stats(sized)
                    min_n = int(args.min_n) if wname == "may_jul09" else min(int(args.min_n), 8)
                    ok = _ok(st, min_n=min_n, min_day_win=float(args.min_day_win))
                    dual_ok = dual_ok and ok
                    for k, v in st.items():
                        row[f"{wname}_{k}"] = v
                    row[f"{wname}_ok"] = ok
                    # equity extras on discovery
                    if wname == "may_jul09":
                        eq = _equity_stats(pd.DataFrame(sized))
                        row["disc_compound"] = eq.get("compound")
                        row["disc_maxdd"] = eq.get("maxdd")
                        row["disc_trade_win"] = eq.get("trade_win")
                blind_bucket = by_w.get("jul10_23") or []
                sized_b = _portfolio_day(
                    sorted(blind_bucket, key=lambda x: (x["entry_ts"], x["symbol"])),
                    position_frac=float(args.position_frac),
                    max_concurrent=int(args.max_concurrent),
                    cooldown_minutes=float(args.cooldown_minutes),
                )
                eqb = _equity_stats(pd.DataFrame(sized_b))
                row["blind_compound"] = eqb.get("compound")
                row["blind_maxdd"] = eqb.get("maxdd")
                row["blind_trade_win"] = eqb.get("trade_win")
                row["dual_pass"] = bool(dual_ok)
                # month compounds on all sized
                all_sized = _portfolio_day(
                    sorted(raw, key=lambda x: (x["entry_ts"], x["symbol"])),
                    position_frac=float(args.position_frac),
                    max_concurrent=int(args.max_concurrent),
                    cooldown_minutes=float(args.cooldown_minutes),
                )
                months = _month_compounds(pd.DataFrame(all_sized))
                row["may"] = months.get("2026-05")
                row["jun"] = months.get("2026-06")
                row["jul"] = months.get("2026-07")
                score_rows.append(row)
                print(
                    f"lag{lag:g}/sp{sp:g} {pol['name']:40s} "
                    f"n={len(raw):2d} flip={row['frac_flipped']:.2f} "
                    f"disc_cmp={row.get('disc_compound')} "
                    f"blind_cmp={row.get('blind_compound')} "
                    f"dual={row['dual_pass']}",
                    flush=True,
                )

    sb = pd.DataFrame(score_rows)
    sb.to_csv(out / "scoreboard.csv", index=False)
    pd.DataFrame(all_trades).to_csv(out / "trades.csv", index=False)

    causal = sb[sb["causal"] == True].copy()  # noqa: E712
    passed = causal[causal["dual_pass"] == True]  # noqa: E712
    # best causal by disc compound among lag5/sp15 (lenient) and lag3/sp10 (strict)
    def _pick(df: pd.DataFrame, lag: float, sp: float) -> pd.DataFrame:
        return df[(df.lag == lag) & (df.spread == sp)].sort_values(
            "disc_compound", ascending=False, na_position="last"
        )

    verdict = {
        "protocol": "am_pocket_flip_quote_dual",
        "entry": args.entry,
        "n_prepared": len(prepared),
        "note": (
            "opp_opt is non-causal (lookahead); only flip_causal/static are live-shaped. "
            "Episode ret for flips compounds legs: (1+r1)*(1+r2)-1."
        ),
        "dual_pass_causal": passed.to_dict(orient="records") if len(passed) else [],
        "lag5_sp15": _pick(causal, 5.0, 0.15).head(10).to_dict(orient="records"),
        "lag3_sp10": _pick(causal, 3.0, 0.10).head(10).to_dict(orient="records"),
        "optimistic_noncausal": sb[sb["causal"] == False]  # noqa: E712
        .sort_values("disc_compound", ascending=False, na_position="last")
        .head(8)
        .to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")

    cols = [
        c
        for c in [
            "policy",
            "causal",
            "lag",
            "spread",
            "n_raw",
            "frac_flipped",
            "disc_trade_win",
            "disc_maxdd",
            "disc_compound",
            "blind_trade_win",
            "blind_compound",
            "may_jul09_ok",
            "jul10_23_ok",
            "dual_pass",
            "may",
            "jun",
            "jul",
        ]
        if c in sb.columns
    ]
    print("\nCAUSAL lag5/sp15", flush=True)
    print(_pick(causal, 5.0, 0.15)[cols].to_string(index=False), flush=True)
    print("\nDUAL PASS (causal)", flush=True)
    print(passed[cols].to_string(index=False) if len(passed) else "(none)", flush=True)
    print("\nOPTIMISTIC non-causal", flush=True)
    print(
        sb[sb["causal"] == False][cols].sort_values("disc_compound", ascending=False).head(8).to_string(index=False)  # noqa: E712
        if len(sb[sb["causal"] == False])  # noqa: E712
        else "(none)",
        flush=True,
    )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
