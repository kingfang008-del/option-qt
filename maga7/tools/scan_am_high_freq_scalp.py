#!/usr/bin/env python3
"""High-frequency AM scalp scan: many trades/day, high win, ~5% mean OK.

Objective shift from sparse high-capture:
  - target tens of trades per day
  - prefer high trade_win
  - mean option ret around +3–6% acceptable
  - dual-window compound > 0

Uses full foresight probe grid (not vd_soft pocket), trade-last marks,
fast TP/SL, low cooldown, multi-symbol concurrent.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_high_freq_scalp \\
    --tag research_am_high_freq_scalp
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
from maga7.tools.scan_am_pocket_regime_ladder_v2 import _window_of
from maga7.tools.scan_am_pocket_risk_optimize import _equity_stats
from maga7.tools.scan_session_horizon_foresight import _paths_by_ticker

DEFAULT_PROBES = Path(
    "/mnt/s990/data/maga7/results/research_am_vwap_foresight_map_may_jul/probes.csv"
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
BDAYS = 60  # May1–Jul23 approx


def _f(r: pd.Series, k: str, default: float = float("nan")) -> float:
    try:
        v = float(r.get(k, default))
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _aligned(r: pd.Series) -> bool:
    fo = _f(r, "from_open_px")
    d = str(r["dir"])
    if not np.isfinite(fo):
        return False
    return (d == "UP" and fo >= 0) or (d == "DN" and fo < 0)


def build_entry_gates() -> list[tuple[str, Callable[[pd.Series], bool]]]:
    def any_align(r: pd.Series) -> bool:
        return _aligned(r)

    def mild_fo(r: pd.Series) -> bool:
        if not _aligned(r):
            return False
        fo = abs(_f(r, "from_open_px"))
        return 0.001 <= fo <= 0.025

    def fo_band(r: pd.Series) -> bool:
        if not _aligned(r):
            return False
        fo = abs(_f(r, "from_open_px"))
        return 0.002 <= fo <= 0.015

    def vwap_soft(r: pd.Series) -> bool:
        if not mild_fo(r):
            return False
        vd = abs(_f(r, "vwap_diff"))
        return 0.0005 <= vd <= 0.012

    def fo_vwap_agree(r: pd.Series) -> bool:
        if not mild_fo(r):
            return False
        fo = _f(r, "from_open_px")
        fv = _f(r, "fo_vwap20")
        if not np.isfinite(fv):
            return False
        return (fo >= 0 and fv >= 0) or (fo < 0 and fv < 0)

    def accel_nonneg(r: pd.Series) -> bool:
        if not mild_fo(r):
            return False
        a = _f(r, "accel_10_30")
        d = str(r["dir"])
        s = a if d == "UP" else -a
        return np.isfinite(s) and s >= 0

    def scalp3(r: pd.Series) -> bool:
        """fo band + vwap soft + fo_vwap agree."""
        return fo_band(r) and vwap_soft(r) and fo_vwap_agree(r)

    def scalp2(r: pd.Series) -> bool:
        return fo_band(r) and fo_vwap_agree(r)

    def am_a_only(r: pd.Series) -> bool:
        return mild_fo(r) and str(r.get("session", "")).startswith("AM_A")

    def every5_align(r: pd.Series) -> bool:
        """Aligned + tod minute % 5 == 0 (already 5m buckets mostly)."""
        return _aligned(r)

    return [
        ("any_align", any_align),
        ("mild_fo", mild_fo),
        ("fo_band", fo_band),
        ("vwap_soft", vwap_soft),
        ("fo_vwap_agree", fo_vwap_agree),
        ("accel_nonneg", accel_nonneg),
        ("scalp2", scalp2),
        ("scalp3", scalp3),
        ("am_a_mild", am_a_only),
        ("every5_align", every5_align),
    ]


def exit_grid() -> list[dict[str, Any]]:
    cfgs = []
    for tp in (0.03, 0.04, 0.05, 0.06, 0.08):
        for sl in (0.04, 0.05, 0.06, 0.08, 0.10):
            for h in (30, 60, 90, 120, 180):
                if sl < tp * 0.6:
                    continue  # avoid tiny SL vs TP asymmetry extremes later via score
                cfgs.append(
                    {
                        "name": f"tp{tp:g}_sl{sl:g}_h{h}",
                        "tp": tp,
                        "sl": sl,
                        "max_hold": h,
                    }
                )
    return cfgs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", default=str(DEFAULT_PROBES))
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--tag", default="research_am_high_freq_scalp")
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--position-frac", type=float, default=0.05)
    ap.add_argument("--max-concurrent", type=int, default=8)
    ap.add_argument("--cooldown-minutes", type=float, default=0.0)
    ap.add_argument("--target-tpd", type=float, default=20.0, help="min trades/bday for 'dense'")
    ap.add_argument("--target-win", type=float, default=0.55)
    ap.add_argument("--target-mean", type=float, default=0.03)
    ap.add_argument(
        "--entries",
        default="mild_fo,fo_band,vwap_soft,fo_vwap_agree,scalp2,scalp3,am_a_mild,accel_nonneg",
    )
    ap.add_argument(
        "--exits",
        default="auto",
        help="comma names or 'auto' for focused ~5% grid",
    )
    ap.add_argument("--max-combos", type=int, default=120)
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    out = Path(prof["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    probes = pd.read_csv(args.probes)
    probes["entry_ts"] = pd.to_datetime(probes["entry_ts"])
    probes["date"] = probes["date"].astype(str)
    probes["window"] = probes["date"].map(lambda d: _window_of(d))
    probes = probes[probes["window"].notna()].copy()
    print(f"probes={len(probes)}", flush=True)

    gmap = dict(build_entry_gates())
    want_e = [x.strip() for x in str(args.entries).split(",") if x.strip()]
    entries = [(n, gmap[n]) for n in want_e if n in gmap]
    if str(args.exits) == "auto":
        # focused grid around ~5%
        exits = []
        for tp in (0.04, 0.05, 0.06):
            for sl in (0.05, 0.06, 0.08, 0.10):
                for h in (45, 60, 90, 120):
                    exits.append({"name": f"tp{tp:g}_sl{sl:g}_h{h}", "tp": tp, "sl": sl, "max_hold": h})
        # a few tighter / looser
        for tp, sl, h in ((0.03, 0.05, 60), (0.05, 0.05, 60), (0.08, 0.08, 90), (0.05, 0.08, 180)):
            exits.append({"name": f"tp{tp:g}_sl{sl:g}_h{h}", "tp": tp, "sl": sl, "max_hold": h})
    else:
        full = {c["name"]: c for c in exit_grid()}
        exits = [full[n] for n in str(args.exits).split(",") if n in full]

    # dedupe exits
    seen = set()
    uniq_ex = []
    for e in exits:
        if e["name"] in seen:
            continue
        seen.add(e["name"])
        uniq_ex.append(e)
    exits = uniq_ex

    # Pre-filter probe indices per entry
    entry_idx: dict[str, np.ndarray] = {}
    for ename, efn in entries:
        m = probes.apply(efn, axis=1).to_numpy()
        idx = np.flatnonzero(m)
        entry_idx[ename] = idx
        print(f"  entry {ename}: n={len(idx)} (~{len(idx)/BDAYS:.1f}/bday raw)", flush=True)

    trades_root = Path(args.trades_root)
    path_cache: dict[tuple[str, str], dict] = {}

    def paths_for(date: str, sym: str):
        key = (date, sym)
        if key not in path_cache:
            tday = load_option_trades(trades_root, sym, date)
            path_cache[key] = (
                _paths_by_ticker(tday) if tday is not None and not tday.empty else {}
            )
        return path_cache[key]

    # Precompute sim results for union of all entry probes × all exits would be huge.
    # Instead: for each entry, take probes, sim each exit (cache path sims by (i, exit)).
    # First build unique probe list for union.
    union = sorted(set(int(i) for idx in entry_idx.values() for i in idx))
    print(f"union probes to sim={len(union)} exits={len(exits)}", flush=True)

    # Simulate all exits for union probes once
    # store: list aligned with union order
    sim_cache: dict[str, list[dict | None]] = {e["name"]: [None] * len(union) for e in exits}
    u_pos = {i: p for p, i in enumerate(union)}

    for j, ii in enumerate(union):
        r = probes.iloc[ii]
        date, sym = str(r["date"]), str(r["symbol"])
        ticker = str(r["ticker"]).replace("O:", "")
        arr = paths_for(date, sym).get(ticker)
        et = to_ny(pd.Timestamp(r["entry_ts"]))
        if arr is None:
            continue
        pts, plast = arr[0], arr[1]
        for ecfg in exits:
            sim = simulate_trade_tpsl(
                pts,
                plast,
                et,
                tp=float(ecfg["tp"]),
                sl=float(ecfg["sl"]),
                max_hold_sec=int(ecfg["max_hold"]),
                slip=float(args.slip),
            )
            if sim is None:
                continue
            sim_cache[ecfg["name"]][j] = {
                "date": date,
                "symbol": sym,
                "dir": str(r["dir"]),
                "session": str(r.get("session", "")),
                "entry_ts": et,
                "exit_ts": et + pd.Timedelta(seconds=float(sim["hold_sec"])),
                "ticker": ticker,
                "ret": float(sim["ret"]),
                "exit_reason": str(sim["reason"]),
                "hold_sec": float(sim["hold_sec"]),
                "oracle_ret": float(r["oracle_ret"]),
                "window": str(r["window"]),
            }
        if (j + 1) % 2000 == 0:
            print(f"  sim {j+1}/{len(union)}", flush=True)

    print("sims done; scoring combos", flush=True)

    rows = []
    combos = [(en, ex) for en, _ in entries for ex in exits]
    if len(combos) > int(args.max_combos):
        # prioritize scalp entries × all exits, then others × subset
        pass  # keep all unless huge

    for ename, _ in entries:
        idxs = entry_idx[ename]
        positions = [u_pos[int(i)] for i in idxs if int(i) in u_pos]
        for ecfg in exits:
            raw = []
            for p in positions:
                tr = sim_cache[ecfg["name"]][p]
                if tr is not None:
                    raw.append(tr)
            # portfolio per window
            win_stats = {}
            sized_all = []
            for wname, _, _ in WINDOWS:
                wr = [t for t in raw if t["window"] == wname]
                by_d: dict[str, list] = {}
                for t in wr:
                    by_d.setdefault(t["date"], []).append(t)
                sized = []
                for _, rs in sorted(by_d.items()):
                    sized.extend(
                        _portfolio_day(
                            sorted(rs, key=lambda x: (x["entry_ts"], x["symbol"])),
                            position_frac=float(args.position_frac),
                            max_concurrent=int(args.max_concurrent),
                            cooldown_minutes=float(args.cooldown_minutes),
                        )
                    )
                ste = _equity_stats(pd.DataFrame(sized)) if sized else {"n": 0, "compound": 0.0, "trade_win": None, "mean": None}
                win_stats[wname] = ste
                sized_all.extend(sized)

            n = len(sized_all)
            if n == 0:
                continue
            rr = np.array([t["ret"] for t in sized_all], dtype=float)
            disc = float(win_stats["may_jul09"].get("compound") or 0)
            blind = float(win_stats["jul10_23"].get("compound") or 0)
            n_d = int(win_stats["may_jul09"].get("n") or 0)
            n_b = int(win_stats["jul10_23"].get("n") or 0)
            tpd = n / float(BDAYS)
            tw = float((rr > 0).mean())
            mean_ret = float(rr.mean())
            med_ret = float(np.median(rr))
            active_days = len({t["date"] for t in sized_all})
            tpd_active = n / max(active_days, 1)
            econ = bool(n_d >= 30 and n_b >= 8 and disc > 0 and blind > 0)
            dense_ok = bool(
                tpd >= float(args.target_tpd)
                and tw >= float(args.target_win)
                and mean_ret >= float(args.target_mean)
                and econ
            )
            # softer: dense-ish
            dense_soft = bool(tpd >= 10 and tw >= 0.52 and mean_ret >= 0.02 and econ)
            rows.append(
                {
                    "entry": ename,
                    "exit": ecfg["name"],
                    "n": n,
                    "tpd": tpd,
                    "tpd_active": tpd_active,
                    "active_days": active_days,
                    "trade_win": tw,
                    "mean_ret": mean_ret,
                    "med_ret": med_ret,
                    "disc_compound": disc,
                    "blind_compound": blind,
                    "disc_n": n_d,
                    "blind_n": n_b,
                    "disc_maxdd": float(win_stats["may_jul09"].get("maxdd") or 0),
                    "econ_dual": econ,
                    "dense_ok": dense_ok,
                    "dense_soft": dense_soft,
                    "frac_tp": float(np.mean([t["exit_reason"] == "tp" for t in sized_all])),
                    "hold_p50": float(np.median([t["hold_sec"] for t in sized_all])),
                }
            )

    sb = pd.DataFrame(rows)
    sb.to_csv(out / "scoreboard.csv", index=False)

    # rank
    ok = sb[sb.econ_dual == True].copy()  # noqa: E712
    ok["score"] = (
        ok["tpd"].clip(upper=40) / 40.0 * 0.35
        + ok["trade_win"] * 0.35
        + (ok["mean_ret"].clip(lower=0, upper=0.08) / 0.08) * 0.20
        + (ok["disc_compound"].clip(lower=0, upper=2) / 2) * 0.10
    )
    ok = ok.sort_values(["dense_ok", "dense_soft", "score", "tpd"], ascending=[False, False, False, False])
    ok.to_csv(out / "ranked.csv", index=False)

    hit = ok[ok.dense_ok == True]  # noqa: E712
    soft = ok[ok.dense_soft == True]  # noqa: E712
    best = ok.iloc[0].to_dict() if len(ok) else None

    # frontier: max tpd among econ & mean>=3% & win>=55%
    fr = ok[(ok.mean_ret >= 0.03) & (ok.trade_win >= 0.55)].sort_values("tpd", ascending=False)
    fr2 = ok[(ok.mean_ret >= 0.05) & (ok.trade_win >= 0.55)].sort_values("tpd", ascending=False)

    promote = "NONE"
    if len(hit):
        promote = f"HF_{hit.iloc[0]['entry']}__{hit.iloc[0]['exit']}"
    elif len(soft):
        promote = f"HF_SOFT_{soft.iloc[0]['entry']}__{soft.iloc[0]['exit']}"
    elif best:
        promote = f"HF_BEST_{best['entry']}__{best['exit']}"

    summary = {
        "protocol": "am_high_freq_scalp",
        "objective": "high_n_high_win_mean_~5pct",
        "portfolio": {
            "position_frac": float(args.position_frac),
            "max_concurrent": int(args.max_concurrent),
            "cooldown_minutes": float(args.cooldown_minutes),
        },
        "targets": {
            "tpd": float(args.target_tpd),
            "win": float(args.target_win),
            "mean": float(args.target_mean),
        },
        "n_combos": int(len(sb)),
        "n_dense_ok": int(len(hit)),
        "n_dense_soft": int(len(soft)),
        "promote": promote,
        "best": best,
        "max_tpd_mean3_win55": fr.head(5).to_dict(orient="records") if len(fr) else [],
        "max_tpd_mean5_win55": fr2.head(5).to_dict(orient="records") if len(fr2) else [],
        "top15": ok.head(15).to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    cols = [
        "entry", "exit", "tpd", "trade_win", "mean_ret", "med_ret",
        "disc_compound", "blind_compound", "n", "dense_ok", "dense_soft", "hold_p50", "frac_tp",
    ]
    print("\n=== TOP (econ dual) ===", flush=True)
    print(ok[cols].head(15).to_string(index=False), flush=True)
    print("\n=== max tpd | mean≥3% win≥55% ===", flush=True)
    print(fr[cols].head(8).to_string(index=False) if len(fr) else "(none)", flush=True)
    print("\n=== max tpd | mean≥5% win≥55% ===", flush=True)
    print(fr2[cols].head(8).to_string(index=False) if len(fr2) else "(none)", flush=True)
    print(json.dumps({"promote": promote, "n_dense_ok": int(len(hit)), "n_dense_soft": int(len(soft))}, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
