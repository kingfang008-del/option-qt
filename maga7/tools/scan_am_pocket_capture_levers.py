#!/usr/bin/env python3
"""Try capture levers one-by-one; keep what dual-window passes.

Levers (sequential, each builds on keepers):
  L1  higher-MFE entry concentration (causal gates / FO / score)
  L2  scale-out (partial take + runner trail/TP)
  L3  option-side peak/trail + ratchet floors

Baseline book: ``no_b_up`` pockets, trade-last slip 1%, 20%/max5/cd10.
Target: mean_capture ≥ 0.20 with econ dual (disc & blind compound > 0).

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_pocket_capture_levers \\
    --tag research_am_pocket_capture_levers
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

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
from maga7.tools.scan_am_pocket_exit_design import _path_window, simulate_exit
from maga7.tools.scan_am_pocket_hold_feature_exit import simulate_hold_features
from maga7.tools.scan_am_pocket_multi_gate import build_gates
from maga7.tools.scan_am_pocket_regime_ladder_v2 import _window_of
from maga7.tools.scan_am_pocket_risk_optimize import POCKET_SETS, _equity_stats, _signed
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
CAPTURE_TARGET = 0.20

# Best research ride from hold_feature_exit
RIDE_COMBO = {
    "name": "ride_combo_st15_f03_tp50",
    "kind": "feat",
    "mode": "combo",
    "stall_sec": 15,
    "look_sec": 10,
    "mom_cut": -0.0005,
    "opt_give": 0.03,
    "arm": 0.05,
    "floor": 0.03,
    "tp2": 0.50,
    "sl": 0.18,
    "min_hold": 15,
    "max_hold": 900,
    "tp": 9.0,
}


def _fo_abs(r: pd.Series) -> float:
    v = r.get("from_open_px", r.get("from_open", np.nan))
    try:
        return abs(float(v))
    except (TypeError, ValueError):
        return float("nan")


def _entry_variants() -> list[tuple[str, Callable[[pd.Series], bool]]]:
    gmap = dict(build_gates())

    def wrap(name: str) -> Callable[[pd.Series], bool]:
        fn = gmap[name]
        return fn

    out: list[tuple[str, Callable[[pd.Series], bool]]] = [
        ("vd_soft", wrap("vd_soft")),
        ("champ", wrap("vd+cont60+mf100+volr12")),
        ("struct4v", wrap("struct4v")),
        ("struct5", wrap("struct5")),
        ("vd+agree+cont60+mf100", wrap("vd+agree+cont60+mf100")),
        ("vd+score5", wrap("vd+score5")),
        ("vd+score6", wrap("vd+score6")),
        ("score5", wrap("score5")),
        ("score6", wrap("score6")),
    ]
    # FO magnitude overlays on vd_soft
    vd = gmap["vd_soft"]
    for thr in (0.003, 0.005, 0.008, 0.010, 0.012):
        def _mk(t: float) -> Callable[[pd.Series], bool]:
            def fn(r: pd.Series) -> bool:
                return bool(vd(r) and _fo_abs(r) >= t)

            return fn

        out.append((f"vd_fo{_fo_tag(thr)}", _mk(thr)))

    # accel overlays
    for amin in (0.0, 0.001, 0.002):
        def _mk_a(a: float) -> Callable[[pd.Series], bool]:
            def fn(r: pd.Series) -> bool:
                try:
                    acc = float(r.get("accel_10_30", np.nan))
                except (TypeError, ValueError):
                    acc = float("nan")
                # signed: for UP want +accel, DN want -accel → use raw * dir sign
                d = str(r.get("dir", "UP"))
                s_acc = acc if d == "UP" else (-acc if np.isfinite(acc) else float("nan"))
                return bool(vd(r) and np.isfinite(s_acc) and s_acc >= a)

            return fn

        out.append((f"vd_acc{a_tag(amin)}", _mk_a(amin)))

    # cont + fo
    cont = gmap["cont60"]
    for thr in (0.005, 0.008):
        def _mk_c(t: float) -> Callable[[pd.Series], bool]:
            def fn(r: pd.Series) -> bool:
                return bool(vd(r) and cont(r) and _fo_abs(r) >= t)

            return fn

        out.append((f"vd_cont_fo{_fo_tag(thr)}", _mk_c(thr)))

    # drop duplicates by name
    seen = set()
    uniq = []
    for n, f in out:
        if n in seen:
            continue
        seen.add(n)
        uniq.append((n, f))
    return uniq


def _fo_tag(thr: float) -> str:
    return f"{int(round(thr * 10000))}bp"


def a_tag(a: float) -> str:
    return f"{int(round(a * 10000))}bp"


def _score_gate_names(gmap: dict) -> None:
    """Ensure score gates exist in build_gates; no-op helper for typing."""
    return None


def simulate_ratchet(
    rets: np.ndarray,
    holds: np.ndarray,
    *,
    sl: float,
    max_hold: float,
    arms: list[tuple[float, float]],
    trail: float,
    trail_arm: float,
    hard_tp: float = 9.0,
) -> dict[str, Any]:
    """Option-only ratchet: after each arm level, raise floor; trail after trail_arm."""
    peak = -1.0
    floor = -float(sl)
    armed_trail = False
    arms_sorted = sorted(arms, key=lambda x: x[0])
    hit = set()

    for i in range(1, len(rets)):
        r = float(rets[i])
        h = float(holds[i])
        if h > max_hold:
            return {"ret": float(rets[i - 1]), "reason": "max_hold", "hold_sec": float(holds[i - 1])}
        peak = max(peak, r)
        for ai, (lvl, fl) in enumerate(arms_sorted):
            if ai not in hit and r >= lvl:
                hit.add(ai)
                floor = max(floor, fl)
        if r >= trail_arm:
            armed_trail = True
        if r >= hard_tp:
            return {"ret": r, "reason": "tp", "hold_sec": h}
        if r <= floor:
            return {"ret": r, "reason": "floor", "hold_sec": h}
        if armed_trail and (peak - r) >= trail:
            return {"ret": r, "reason": "trail", "hold_sec": h}
    return {"ret": float(rets[-1]), "reason": "max_hold", "hold_sec": float(holds[-1])}


def _eval_book(
    bundles: list[dict],
    *,
    exit_fn: Callable[[dict], dict[str, Any] | None],
    position_frac: float,
    max_concurrent: int,
    cooldown_minutes: float,
) -> dict[str, Any]:
    win_raw: dict[str, list] = {w[0]: [] for w in WINDOWS}
    for b in bundles:
        sim = exit_fn(b)
        if sim is None:
            continue
        ret = float(sim["ret"])
        hold = float(sim["hold_sec"])
        if not np.isfinite(ret):
            continue
        win_raw[b["window"]].append(
            {
                "date": b["date"],
                "symbol": b["symbol"],
                "dir": b["dir"],
                "session": b["session"],
                "entry_ts": b["et"],
                "exit_ts": b["et"] + pd.Timedelta(seconds=hold),
                "ticker": b["ticker"],
                "ret": ret,
                "exit_reason": str(sim.get("reason", "")),
                "hold_sec": hold,
                "oracle_ret": b["oracle"],
                "window": b["window"],
            }
        )
    sized_all = []
    win_stats = {}
    for wname, _, _ in WINDOWS:
        by_d: dict[str, list] = {}
        for tr in win_raw[wname]:
            by_d.setdefault(str(tr["date"]), []).append(tr)
        sized = []
        for _, rs in sorted(by_d.items()):
            sized.extend(
                _portfolio_day(
                    sorted(rs, key=lambda x: (x["entry_ts"], x["symbol"])),
                    position_frac=position_frac,
                    max_concurrent=max_concurrent,
                    cooldown_minutes=cooldown_minutes,
                )
            )
        ste = _equity_stats(pd.DataFrame(sized)) if sized else {"n": 0, "compound": 0.0}
        win_stats[wname] = ste
        sized_all.extend(sized)
    disc = float(win_stats["may_jul09"].get("compound") or 0)
    blind = float(win_stats["jul10_23"].get("compound") or 0)
    n_d = int(win_stats["may_jul09"].get("n") or 0)
    n_b = int(win_stats["jul10_23"].get("n") or 0)
    if sized_all:
        o = np.array([t["oracle_ret"] for t in sized_all], dtype=float)
        rr = np.array([t["ret"] for t in sized_all], dtype=float)
        cap = float(rr.mean() / o.mean()) if np.nanmean(o) > 0 else float("nan")
        mean_ret = float(rr.mean())
        mean_oracle = float(np.nanmean(o))
        tw = float((rr > 0).mean())
    else:
        cap = mean_ret = mean_oracle = tw = float("nan")
    econ = bool(n_d >= 6 and n_b >= 3 and disc > 0 and blind > 0)
    return {
        "n": len(sized_all),
        "disc_n": n_d,
        "blind_n": n_b,
        "disc_compound": disc,
        "blind_compound": blind,
        "mean_capture": cap,
        "mean_ret": mean_ret,
        "mean_oracle": mean_oracle,
        "trade_win": tw,
        "econ_dual": econ,
        "hit_capture_target": bool(np.isfinite(cap) and cap >= CAPTURE_TARGET),
        "eq_maxdd": float(win_stats["may_jul09"].get("maxdd") or 0),
    }


def _exit_tp8(b: dict, slip: float) -> dict[str, Any] | None:
    sim = simulate_trade_tpsl(
        b["pts"], b["plast"], b["et"], tp=0.08, sl=0.15, max_hold_sec=240, slip=slip
    )
    if sim is None:
        return None
    return {"ret": float(sim["ret"]), "hold_sec": float(sim["hold_sec"]), "reason": str(sim["reason"])}


def _exit_ride(b: dict) -> dict[str, Any] | None:
    # ride_combo needs stock arrays — skip if missing
    if b.get("sarr") is None:
        return None
    return simulate_hold_features(
        b["rets"],
        b["holds"],
        stock_arr=b["sarr"],
        entry_ts=b["et"],
        direction=b["dir"],
        mode=str(RIDE_COMBO["mode"]),
        params=RIDE_COMBO,
    )


def _exit_cfg(b: dict, cfg: dict[str, Any], slip: float) -> dict[str, Any] | None:
    kind = cfg["kind"]
    if kind == "tpsl":
        return _exit_tp8(b, slip)
    if kind == "ride":
        return _exit_ride(b)
    if kind == "trail":
        return simulate_exit(b["rets"], b["holds"], mode="trail", params=cfg)
    if kind == "hybrid":
        return simulate_exit(b["rets"], b["holds"], mode="hybrid", params=cfg)
    if kind == "scale":
        return simulate_scaleout(
            b["rets"],
            b["holds"],
            frac1=float(cfg["frac1"]),
            tp1=float(cfg["tp1"]),
            sl=float(cfg["sl"]),
            max_hold=float(cfg["max_hold"]),
            runner=str(cfg["runner"]),
            tp2=float(cfg.get("tp2", 9.0)),
            arm=float(cfg.get("arm", 0.0)),
            trail=float(cfg.get("trail", 9.0)),
            floor=float(cfg.get("floor", 0.0)),
            be_after_scale=bool(cfg.get("be_after_scale", False)),
        )
    if kind == "ratchet":
        return simulate_ratchet(
            b["rets"],
            b["holds"],
            sl=float(cfg["sl"]),
            max_hold=float(cfg["max_hold"]),
            arms=list(cfg["arms"]),
            trail=float(cfg["trail"]),
            trail_arm=float(cfg["trail_arm"]),
            hard_tp=float(cfg.get("hard_tp", 9.0)),
        )
    if kind == "feat":
        if b.get("sarr") is None:
            return None
        return simulate_hold_features(
            b["rets"],
            b["holds"],
            stock_arr=b["sarr"],
            entry_ts=b["et"],
            direction=b["dir"],
            mode=str(cfg["mode"]),
            params=cfg,
        )
    raise ValueError(kind)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--enriched", default=str(DEFAULT_ENRICHED))
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--stock-1s", default="/mnt/s990/data/raw_1s/stocks")
    ap.add_argument("--tag", default="research_am_pocket_capture_levers")
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--position-frac", type=float, default=0.20)
    ap.add_argument("--max-concurrent", type=int, default=5)
    ap.add_argument("--cooldown-minutes", type=float, default=10.0)
    args = ap.parse_args(argv)

    from maga7.common.bar_agg import load_stock_1s_day
    from maga7.common.session_1s_features import prepare_day_arrays

    prof = load_profile(args.profile)
    out = Path(prof["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    probes = pd.read_csv(args.enriched)
    probes["entry_ts"] = pd.to_datetime(probes["entry_ts"])
    probes = probes[probes["enrich_ok"] == True].copy()  # noqa: E712
    pdf = pd.DataFrame(sorted(POCKET_SETS["no_b_up"]), columns=["session", "tod_bucket", "dir"])
    probes = probes.merge(pdf, on=["session", "tod_bucket", "dir"], how="inner")

    gmap = dict(build_gates())
    trades_root = Path(args.trades_root)
    stock_root = Path(args.stock_1s)
    tcache: dict = {}
    scache: dict = {}

    def paths_for(date, sym):
        key = (date, sym)
        if key not in tcache:
            tday = load_option_trades(trades_root, sym, date)
            tcache[key] = _paths_by_ticker(tday) if tday is not None and not tday.empty else {}
        return tcache[key]

    def stock_for(date, sym):
        key = (date, sym)
        if key not in scache:
            raw = load_stock_1s_day(stock_root, sym, date)
            scache[key] = None if raw is None or raw.empty else prepare_day_arrays(raw)
        return scache[key]

    # Build ALL pocket probes with paths once (filter per entry later)
    all_rows = []
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
        pw = _path_window(arr[0], arr[1], et, max_hold_sec=900, slip=float(args.slip))
        if pw is None:
            continue
        rets, holds, _, _ = pw
        sarr = stock_for(date, sym)
        all_rows.append(
            {
                "date": date,
                "symbol": sym,
                "dir": str(r["dir"]),
                "session": str(r["session"]),
                "ticker": ticker,
                "window": w,
                "et": et,
                "rets": rets,
                "holds": holds,
                "pts": arr[0],
                "plast": arr[1],
                "sarr": sarr,
                "oracle": float(r["oracle_ret"]),
                "row": r,
            }
        )
    print(f"path-ready probes={len(all_rows)}", flush=True)

    port = dict(
        position_frac=float(args.position_frac),
        max_concurrent=int(args.max_concurrent),
        cooldown_minutes=float(args.cooldown_minutes),
    )

    def bundles_for(pred: Callable[[pd.Series], bool]) -> list[dict]:
        return [b for b in all_rows if pred(b["row"])]

    # ---------- L1: entry concentration ----------
    print("\n=== L1 entry concentration ===", flush=True)
    l1_rows = []
    entries = _entry_variants()

    for ename, efn in entries:
        bun = bundles_for(efn)
        if len(bun) < 8:
            continue
        for exit_name, ekind in (("tp8", "tpsl"), ("ride", "ride")):
            cfg = {"kind": ekind, "name": exit_name}
            st = _eval_book(
                bun,
                exit_fn=lambda b, c=cfg: _exit_cfg(b, c, float(args.slip)),
                **port,
            )
            # oracle concentration of selected set
            omean = float(np.mean([b["oracle"] for b in bun])) if bun else float("nan")
            row = {
                "lever": "L1",
                "entry": ename,
                "exit": exit_name,
                "n_probes": len(bun),
                "oracle_conc": omean,
                **st,
            }
            l1_rows.append(row)
        print(
            f"  {ename}: n={len(bun)} o={omean:.3f} "
            f"tp8_cap={l1_rows[-2]['mean_capture']:.3f} "
            f"ride_cap={l1_rows[-1]['mean_capture']:.3f} "
            f"ride_econ={l1_rows[-1]['econ_dual']}",
            flush=True,
        )

    l1 = pd.DataFrame(l1_rows)
    l1.to_csv(out / "l1_entries.csv", index=False)
    l1_ok = l1[(l1.econ_dual == True) & (l1.exit == "ride")].copy()  # noqa: E712
    l1_ok = l1_ok.sort_values(["mean_capture", "disc_compound"], ascending=[False, False])
    l1_keep = l1_ok.head(5)
    # also keep best by oracle_conc among econ
    l1_conc = l1_ok.sort_values("oracle_conc", ascending=False).head(3)
    keep_entries = list(dict.fromkeys(
        ["vd_soft"] + l1_keep["entry"].tolist() + l1_conc["entry"].tolist()
    ))
    print(f"L1 keep entries: {keep_entries}", flush=True)
    print(l1_ok[["entry", "n", "oracle_conc", "mean_capture", "disc_compound", "blind_compound"]].head(10).to_string(index=False), flush=True)

    # ---------- L2: scale-out ----------
    print("\n=== L2 scale-out ===", flush=True)
    scale_cfgs: list[dict[str, Any]] = []
    for frac in (0.33, 0.50, 0.67):
        for tp1 in (0.06, 0.08, 0.10, 0.12):
            for trail in (0.06, 0.08, 0.10, 0.12):
                for arm in (0.08, 0.12, 0.15):
                    scale_cfgs.append(
                        {
                            "name": f"sc{frac:g}@{tp1:g}_tr{trail:g}_a{arm:g}",
                            "kind": "scale",
                            "frac1": frac,
                            "tp1": tp1,
                            "sl": 0.15,
                            "max_hold": 900,
                            "runner": "trail",
                            "arm": arm,
                            "trail": trail,
                            "floor": 0.0,
                            "be_after_scale": True,
                        }
                    )
            for tp2 in (0.20, 0.30, 0.40, 0.50):
                scale_cfgs.append(
                    {
                        "name": f"sc{frac:g}@{tp1:g}_tp2{tp2:g}_be",
                        "kind": "scale",
                        "frac1": frac,
                        "tp1": tp1,
                        "sl": 0.15,
                        "max_hold": 900,
                        "runner": "tp",
                        "tp2": tp2,
                        "floor": 0.0,
                        "be_after_scale": True,
                        "arm": tp1,
                    }
                )
    # trim: too many — subsample by stride if huge
    if len(scale_cfgs) > 180:
        scale_cfgs = scale_cfgs[::2]
    print(f"scale cfgs={len(scale_cfgs)} × entries={len(keep_entries)}", flush=True)

    l2_rows = []
    efn_map = dict(_entry_variants())
    for sn in gmap:
        if sn not in efn_map:
            efn_map[sn] = gmap[sn]
    for ename in keep_entries:
        efn = efn_map.get(ename)
        if efn is None:
            continue
        bun = bundles_for(efn)
        if len(bun) < 8:
            continue
        for cfg in scale_cfgs:
            st = _eval_book(
                bun,
                exit_fn=lambda b, c=cfg: _exit_cfg(b, c, float(args.slip)),
                **port,
            )
            l2_rows.append({"lever": "L2", "entry": ename, "exit": cfg["name"], **st})
        # progress
        best_e = max(
            (r for r in l2_rows if r["entry"] == ename),
            key=lambda r: (r["econ_dual"], r["mean_capture"] if np.isfinite(r["mean_capture"]) else -1),
        )
        print(
            f"  {ename}: best_cap={best_e['mean_capture']:.3f} "
            f"econ={best_e['econ_dual']} exit={best_e['exit']}",
            flush=True,
        )

    l2 = pd.DataFrame(l2_rows)
    l2.to_csv(out / "l2_scaleout.csv", index=False)
    l2_ok = l2[l2.econ_dual == True].copy()  # noqa: E712
    l2_ok = l2_ok.sort_values(["mean_capture", "disc_compound"], ascending=[False, False])
    print("L2 top:", flush=True)
    print(l2_ok[["entry", "exit", "n", "mean_capture", "disc_compound", "blind_compound"]].head(12).to_string(index=False), flush=True)

    # ---------- L3: option peak/trail + ratchet ----------
    print("\n=== L3 option peak/trail ===", flush=True)
    l3_cfgs: list[dict[str, Any]] = []
    for arm in (0.06, 0.08, 0.10, 0.12, 0.15):
        for trail in (0.04, 0.06, 0.08, 0.10, 0.12):
            for tp in (0.25, 0.40, 0.60, 9.0):
                l3_cfgs.append(
                    {
                        "name": f"tr_a{arm:g}_t{trail:g}_tp{tp:g}",
                        "kind": "trail",
                        "arm": arm,
                        "trail": trail,
                        "tp": tp,
                        "sl": 0.15,
                        "max_hold": 900,
                    }
                )
                l3_cfgs.append(
                    {
                        "name": f"hy_a{arm:g}_t{trail:g}_f0_tp{tp:g}",
                        "kind": "hybrid",
                        "arm": arm,
                        "trail": trail,
                        "floor": 0.0,
                        "tp": tp,
                        "sl": 0.15,
                        "max_hold": 900,
                    }
                )
    # ratchet floors
    ratchets = [
        {"name": "rat_08be_15p5_tr08", "arms": [(0.08, 0.0), (0.15, 0.05)], "trail": 0.08, "trail_arm": 0.12},
        {"name": "rat_06be_12p3_20p8_tr06", "arms": [(0.06, 0.0), (0.12, 0.03), (0.20, 0.08)], "trail": 0.06, "trail_arm": 0.10},
        {"name": "rat_08be_20p8_30p15_tr10", "arms": [(0.08, 0.0), (0.20, 0.08), (0.30, 0.15)], "trail": 0.10, "trail_arm": 0.15},
        {"name": "rat_10be_20p5_tr08", "arms": [(0.10, 0.0), (0.20, 0.05)], "trail": 0.08, "trail_arm": 0.12},
        {"name": "rat_08be_15p5_25p12_tr05", "arms": [(0.08, 0.0), (0.15, 0.05), (0.25, 0.12)], "trail": 0.05, "trail_arm": 0.10},
        {"name": "rat_05be_10p2_20p8_tr04", "arms": [(0.05, 0.0), (0.10, 0.02), (0.20, 0.08)], "trail": 0.04, "trail_arm": 0.08},
    ]
    for rc in ratchets:
        for hard_tp in (0.40, 0.60, 9.0):
            l3_cfgs.append(
                {
                    "name": f"{rc['name']}_htp{hard_tp:g}",
                    "kind": "ratchet",
                    "arms": rc["arms"],
                    "trail": rc["trail"],
                    "trail_arm": rc["trail_arm"],
                    "sl": 0.15,
                    "max_hold": 900,
                    "hard_tp": hard_tp,
                }
            )
    if len(l3_cfgs) > 220:
        l3_cfgs = l3_cfgs[::2]
    print(f"l3 cfgs={len(l3_cfgs)}", flush=True)

    # Use top 3 L1 entries by capture for L3
    top_entries = keep_entries[:4]
    l3_rows = []
    for ename in top_entries:
        efn = efn_map.get(ename)
        if efn is None:
            continue
        bun = bundles_for(efn)
        if len(bun) < 8:
            continue
        for cfg in l3_cfgs:
            st = _eval_book(
                bun,
                exit_fn=lambda b, c=cfg: _exit_cfg(b, c, float(args.slip)),
                **port,
            )
            l3_rows.append({"lever": "L3", "entry": ename, "exit": cfg["name"], **st})
        best_e = max(
            (r for r in l3_rows if r["entry"] == ename),
            key=lambda r: (r["econ_dual"], r["mean_capture"] if np.isfinite(r["mean_capture"]) else -1),
        )
        print(
            f"  {ename}: best_cap={best_e['mean_capture']:.3f} "
            f"econ={best_e['econ_dual']} exit={best_e['exit']}",
            flush=True,
        )

    l3 = pd.DataFrame(l3_rows)
    l3.to_csv(out / "l3_opt_trail.csv", index=False)
    l3_ok = l3[l3.econ_dual == True].copy()  # noqa: E712
    l3_ok = l3_ok.sort_values(["mean_capture", "disc_compound"], ascending=[False, False])
    print("L3 top:", flush=True)
    print(l3_ok[["entry", "exit", "n", "mean_capture", "disc_compound", "blind_compound"]].head(12).to_string(index=False), flush=True)

    # ---------- combine keepers ----------
    print("\n=== COMBINED keepers ===", flush=True)
    all_ok = pd.concat(
        [
            l1[l1.econ_dual == True],  # noqa: E712
            l2_ok,
            l3_ok,
        ],
        ignore_index=True,
    )
    all_ok = all_ok.sort_values(
        ["hit_capture_target", "mean_capture", "disc_compound"],
        ascending=[False, False, False],
    )
    all_ok.to_csv(out / "all_econ_ranked.csv", index=False)

    hit = all_ok[all_ok.hit_capture_target == True]  # noqa: E712
    best = all_ok.iloc[0].to_dict() if len(all_ok) else None
    best_per_lever = {}
    for lev in ("L1", "L2", "L3"):
        sub = all_ok[all_ok.lever == lev]
        if len(sub):
            best_per_lever[lev] = sub.iloc[0].to_dict()

    # vs baselines
    base_tp8 = l1[(l1.entry == "vd_soft") & (l1.exit == "tp8")]
    base_ride = l1[(l1.entry == "vd_soft") & (l1.exit == "ride")]
    baseline = {
        "vd_soft_tp8": base_tp8.iloc[0].to_dict() if len(base_tp8) else {},
        "vd_soft_ride": base_ride.iloc[0].to_dict() if len(base_ride) else {},
    }

    promote = "NONE"
    if len(hit):
        promote = f"CAPTURE20_{hit.iloc[0]['lever']}_{hit.iloc[0]['entry']}__{hit.iloc[0]['exit']}"
    elif best and float(best.get("mean_capture") or 0) > float(baseline.get("vd_soft_ride", {}).get("mean_capture") or 0) + 0.005:
        promote = f"CAPTURE_LIFT_{best['lever']}_{best['entry']}__{best['exit']}"
    elif best:
        promote = f"KEEP_{best['lever']}_{best['entry']}__{best['exit']}"

    keepers = {
        "L1_best": best_per_lever.get("L1"),
        "L2_best": best_per_lever.get("L2"),
        "L3_best": best_per_lever.get("L3"),
        "overall_best": best,
        "n_hit_20": int(len(hit)),
        "promote": promote,
    }

    summary = {
        "protocol": "am_pocket_capture_levers",
        "capture_target": CAPTURE_TARGET,
        "baseline": baseline,
        "keepers": keepers,
        "l1_n": int(len(l1)),
        "l2_n": int(len(l2)),
        "l3_n": int(len(l3)),
        "top15": all_ok.head(15).to_dict(orient="records"),
        "hit20": hit.head(10).to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== VERDICT ===", flush=True)
    for lev, row in best_per_lever.items():
        print(
            f"{lev}: {row['entry']} + {row['exit']} → cap={row['mean_capture']:.3f} "
            f"disc={row['disc_compound']:+.3f} blind={row['blind_compound']:+.3f} "
            f"n={row['n']} hit20={row['hit_capture_target']}",
            flush=True,
        )
    print(f"promote={promote} n_hit20={len(hit)}", flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
