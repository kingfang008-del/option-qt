#!/usr/bin/env python3
"""Grid ``stock_up`` thresholds on widen-only TP2 ladders (trade-mark only).

Fixes entry = vd_soft (champ optional). Ladder pack = LADDERS_V2 (hard tp2).
Sweeps confirm_sec × stock_min × opt_min; ranks vs fixed TP8.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_pocket_stock_up_grid \\
    --tag research_am_pocket_stock_up_grid
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
from maga7.common.option_trade_tpsl import simulate_trade_tpsl
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import to_ny
from maga7.common.session_1s_features import prepare_day_arrays
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_delayed_confirm_quote_dual import _stats
from maga7.tools.scan_am_pocket_exit_design import _path_window
from maga7.tools.scan_am_pocket_multi_gate import build_gates
from maga7.tools.scan_am_pocket_path_exit import _stock_series, _stock_signed_at
from maga7.tools.scan_am_pocket_regime_ladder_v2 import (
    LADDERS_V2,
    _apply_ladder,
    _window_of,
    classify_regime_v2,
)
from maga7.tools.scan_am_pocket_risk_optimize import POCKET_SETS, _equity_stats, _month_compounds
from maga7.tools.scan_session_horizon_foresight import _paths_by_ticker

DEFAULT_ENRICHED = Path(
    "/mnt/s990/data/maga7/results/research_am_pocket_multi_gate/enriched_probes.csv"
)
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")
DEFAULT_STOCK = Path("/mnt/s990/data/raw_1s/stocks")
PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
WINDOWS = (
    ("may_jul09", "2026-05-01", "2026-07-09"),
    ("jul10_23", "2026-07-10", "2026-07-23"),
)


def _portfolio_stats(raw_by_window: dict[str, list[dict]], *, frac: float, maxc: int, cd: float):
    sized_all: list[dict] = []
    stats: dict[str, dict] = {}
    for wname, _, _ in WINDOWS:
        raw = raw_by_window[wname]
        by_d: dict[str, list] = {}
        for tr in raw:
            by_d.setdefault(str(tr["date"]), []).append(tr)
        sized: list[dict] = []
        for _, rs in sorted(by_d.items()):
            sized.extend(
                _portfolio_day(
                    sorted(rs, key=lambda x: (x["entry_ts"], x["symbol"])),
                    position_frac=frac,
                    max_concurrent=maxc,
                    cooldown_minutes=cd,
                )
            )
        st = _stats(sized) if sized else {"n": 0, "mean": None}
        ste = _equity_stats(pd.DataFrame(sized)) if sized else {}
        for k, v in ste.items():
            st[f"eq_{k}"] = v
        stats[wname] = st
        sized_all.extend(sized)
    if sized_all:
        o = np.array([t["oracle_ret"] for t in sized_all], dtype=float)
        rr = np.array([t["ret"] for t in sized_all], dtype=float)
        cap = float(rr.mean() / o.mean()) if np.nanmean(o) > 0 else float("nan")
    else:
        cap = float("nan")
    months = _month_compounds(pd.DataFrame(sized_all)) if sized_all else {}
    disc_c = float(stats["may_jul09"].get("eq_compound") or 0)
    blind_c = float(stats["jul10_23"].get("eq_compound") or 0)
    n_d = int(stats["may_jul09"].get("n") or 0)
    n_b = int(stats["jul10_23"].get("n") or 0)
    return {
        "stats": stats,
        "sized_all": sized_all,
        "mean_capture": cap,
        "disc_compound": disc_c,
        "blind_compound": blind_c,
        "disc_n": n_d,
        "blind_n": n_b,
        "econ_dual": bool(n_d >= 8 and n_b >= 3 and disc_c > 0 and blind_c > 0),
        "may": months.get("2026-05"),
        "jun": months.get("2026-06"),
        "jul": months.get("2026-07"),
        "disc_mean": stats["may_jul09"].get("mean"),
        "blind_mean": stats["jul10_23"].get("mean"),
        "disc_maxdd": stats["may_jul09"].get("eq_maxdd"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--enriched", default=str(DEFAULT_ENRICHED))
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--stock-1s", default=str(DEFAULT_STOCK))
    ap.add_argument("--tag", default="research_am_pocket_stock_up_grid")
    ap.add_argument("--entry", default="vd_soft")
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--position-frac", type=float, default=0.20)
    ap.add_argument("--max-concurrent", type=int, default=5)
    ap.add_argument("--cooldown-minutes", type=float, default=10.0)
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    out = Path(prof["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    probes_all = pd.read_csv(args.enriched)
    probes_all["entry_ts"] = pd.to_datetime(probes_all["entry_ts"])
    probes_all = probes_all[probes_all["enrich_ok"] == True].copy()  # noqa: E712
    pdf = pd.DataFrame(sorted(POCKET_SETS["no_b_up"]), columns=["session", "tod_bucket", "dir"])
    probes_all = probes_all.merge(pdf, on=["session", "tod_bucket", "dir"], how="inner")
    gfn = dict(build_gates())[str(args.entry)]
    probes = probes_all[probes_all.apply(gfn, axis=1)].copy()
    print(f"entry={args.entry} probes={len(probes)}", flush=True)

    trades_root = Path(args.trades_root)
    stock_root = Path(args.stock_1s)
    tcache: dict[tuple[str, str], dict] = {}
    scache: dict[tuple[str, str], Any] = {}

    def paths_for(date: str, sym: str):
        key = (date, sym)
        if key not in tcache:
            tday = load_option_trades(trades_root, sym, date)
            tcache[key] = (
                _paths_by_ticker(tday) if tday is not None and not tday.empty else {}
            )
        return tcache[key]

    def stock_arr(date: str, sym: str):
        key = (date, sym)
        if key not in scache:
            raw = load_stock_1s_day(stock_root, sym, date)
            scache[key] = None if raw is None or raw.empty else prepare_day_arrays(raw)
        return scache[key]

    bundles: list[dict[str, Any]] = []
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
        pw = _path_window(arr[0], arr[1], et, max_hold_sec=600, slip=float(args.slip))
        if pw is None:
            continue
        rets, holds, _, _ = pw
        sarr = stock_arr(date, sym)
        sh = sp = None
        if sarr is not None:
            ss = _stock_series(sarr, et, 600)
            if ss is not None:
                sh, sp = ss
        bundles.append(
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
                "sh": sh,
                "sp": sp,
                "pts": arr[0],
                "plast": arr[1],
                "oracle": float(r["oracle_ret"]),
                "reg0": classify_regime_v2(r),
            }
        )
    print(f"bundles={len(bundles)}", flush=True)

    def eval_upgrade(confirm_sec: float, stock_min: float, opt_min: float, *, use: bool):
        win_raw: dict[str, list] = {w[0]: [] for w in WINDOWS}
        n_up = 0
        for b in bundles:
            reg = b["reg0"]
            tag = None
            if use and b["sh"] is not None and reg != "IMPULSE":
                i = int(np.searchsorted(b["holds"], confirm_sec, side="left"))
                i = min(max(i, 1), len(b["rets"]) - 1)
                r_opt = float(b["rets"][i])
                s = _stock_signed_at(
                    b["sh"], b["sp"], float(b["sp"][0]), b["dir"], float(b["holds"][i])
                )
                if r_opt >= opt_min and s >= stock_min:
                    reg = "IMPULSE"
                    tag = "stock_upgrade"
                    n_up += 1
            sim = _apply_ladder(b["rets"], b["holds"], reg, pack=LADDERS_V2)
            if sim is None or not np.isfinite(sim.get("ret", float("nan"))):
                continue
            hold = float(sim["hold_sec"])
            ret = float(sim["ret"])
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
                    "exit_reason": str(sim["reason"]) + (f"+{tag}" if tag else ""),
                    "hold_sec": hold,
                    "oracle_ret": b["oracle"],
                    "regime": reg,
                }
            )
        m = _portfolio_stats(
            win_raw,
            frac=float(args.position_frac),
            maxc=int(args.max_concurrent),
            cd=float(args.cooldown_minutes),
        )
        m["n_upgrade"] = n_up
        m["n_raw"] = sum(len(v) for v in win_raw.values())
        return m

    rows: list[dict[str, Any]] = []

    # fixed TP8 baseline
    win_raw: dict[str, list] = {w[0]: [] for w in WINDOWS}
    for b in bundles:
        sim = simulate_trade_tpsl(
            b["pts"], b["plast"], b["et"], tp=0.08, sl=0.15, max_hold_sec=240, slip=float(args.slip)
        )
        if sim is None:
            continue
        hold = float(sim["hold_sec"])
        win_raw[b["window"]].append(
            {
                "date": b["date"],
                "symbol": b["symbol"],
                "dir": b["dir"],
                "session": b["session"],
                "entry_ts": b["et"],
                "exit_ts": b["et"] + pd.Timedelta(seconds=hold),
                "ticker": b["ticker"],
                "ret": float(sim["ret"]),
                "exit_reason": str(sim["reason"]),
                "hold_sec": hold,
                "oracle_ret": b["oracle"],
            }
        )
    base_m = _portfolio_stats(
        win_raw,
        frac=float(args.position_frac),
        maxc=int(args.max_concurrent),
        cd=float(args.cooldown_minutes),
    )
    base_row = {
        "name": "fixed_tp8",
        "confirm_sec": None,
        "stock_min": None,
        "opt_min": None,
        "n_upgrade": 0,
        "n_raw": sum(len(v) for v in win_raw.values()),
        **{k: base_m[k] for k in (
            "mean_capture", "disc_compound", "blind_compound",
            "disc_n", "blind_n", "econ_dual", "may", "jun", "jul",
            "disc_mean", "blind_mean", "disc_maxdd",
        )},
    }
    rows.append(base_row)
    print(
        f"BASE fixed_tp8 disc={base_row['disc_compound']:+.3f} "
        f"blind={base_row['blind_compound']:+.3f} cap={base_row['mean_capture']:.4f}",
        flush=True,
    )

    no_up = eval_upgrade(45, 9.0, 9.0, use=False)
    rows.append(
        {
            "name": "widen_tp2_no_up",
            "confirm_sec": None,
            "stock_min": None,
            "opt_min": None,
            **{k: no_up[k] for k in (
                "n_upgrade", "n_raw", "mean_capture", "disc_compound", "blind_compound",
                "disc_n", "blind_n", "econ_dual", "may", "jun", "jul",
                "disc_mean", "blind_mean", "disc_maxdd",
            )},
        }
    )

    # prior default
    prior = eval_upgrade(45, 0.0015, 0.03, use=True)
    rows.append(
        {
            "name": "prior_c45_s15bp_o3",
            "confirm_sec": 45,
            "stock_min": 0.0015,
            "opt_min": 0.03,
            **{k: prior[k] for k in (
                "n_upgrade", "n_raw", "mean_capture", "disc_compound", "blind_compound",
                "disc_n", "blind_n", "econ_dual", "may", "jun", "jul",
                "disc_mean", "blind_mean", "disc_maxdd",
            )},
        }
    )

    grid = [
        (c, s, o)
        for c in (30, 45, 60, 90)
        for s in (0.0005, 0.0010, 0.0015, 0.0020, 0.0030, 0.0050)
        for o in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)
    ]
    print(f"grid cells={len(grid)}", flush=True)
    for i, (c, s, o) in enumerate(grid):
        m = eval_upgrade(c, s, o, use=True)
        rows.append(
            {
                "name": f"c{c}_s{s:g}_o{o:g}",
                "confirm_sec": c,
                "stock_min": s,
                "opt_min": o,
                **{k: m[k] for k in (
                    "n_upgrade", "n_raw", "mean_capture", "disc_compound", "blind_compound",
                    "disc_n", "blind_n", "econ_dual", "may", "jun", "jul",
                    "disc_mean", "blind_mean", "disc_maxdd",
                )},
            }
        )
        if (i + 1) % 36 == 0:
            print(f"[{i+1}/{len(grid)}]", flush=True)

    sb = pd.DataFrame(rows)
    sb.to_csv(out / "scoreboard.csv", index=False)

    base_cap = float(base_row["mean_capture"])
    base_disc = float(base_row["disc_compound"])
    base_blind = float(base_row["blind_compound"])

    ok = sb[sb["econ_dual"] == True].copy()  # noqa: E712
    ok["cap_lift"] = ok["mean_capture"] - base_cap
    ok["disc_lift"] = ok["disc_compound"] - base_disc
    ok["blind_lift"] = ok["blind_compound"] - base_blind
    ok["beat_cap"] = ok["mean_capture"] > base_cap + 1e-6
    ok["beat_disc"] = ok["disc_compound"] > base_disc + 1e-9
    ok["blind_ok"] = ok["blind_compound"] + 1e-9 >= base_blind - 0.02
    ok["score"] = ok["cap_lift"] * 3.0 + ok["disc_lift"] + ok["blind_lift"].clip(lower=-0.05)
    ok = ok.sort_values(
        ["beat_cap", "beat_disc", "score", "mean_capture", "disc_compound"],
        ascending=[False, False, False, False, False],
    )
    ok.to_csv(out / "ranked.csv", index=False)

    both = ok[ok["beat_cap"] & ok["beat_disc"] & ok["blind_ok"] & (ok["name"] != "fixed_tp8")]
    print("\n=== TOP 12 ===", flush=True)
    cols = [
        "name", "confirm_sec", "stock_min", "opt_min", "n_upgrade",
        "mean_capture", "disc_compound", "blind_compound", "cap_lift", "disc_lift", "blind_lift",
    ]
    print(ok[cols].head(12).to_string(index=False), flush=True)
    print(f"\nbeat cap+disc+blind_ok: {len(both)}", flush=True)
    if len(both):
        print(both[cols].head(8).to_string(index=False), flush=True)

    best_both = both.iloc[0].to_dict() if len(both) else None
    best_any = ok[ok["name"] != "fixed_tp8"].iloc[0].to_dict() if len(ok) > 1 else None
    # stability: among both, prefer mid n_upgrade (not 0, not all)
    stable = None
    if len(both):
        b2 = both[(both["n_upgrade"] >= 5) & (both["n_upgrade"] <= 40)].copy()
        if len(b2):
            b2 = b2.sort_values(["score", "mean_capture"], ascending=False)
            stable = b2.iloc[0].to_dict()

    promote = "NONE"
    pick = stable or best_both
    if pick is not None:
        promote = f"STOCK_UP_{pick['name']}"

    summary = {
        "protocol": "am_pocket_stock_up_grid",
        "entry": args.entry,
        "ladder": "LADDERS_V2_hard_tp2",
        "mark": "option_trade_last_slip",
        "base": base_row,
        "n_grid": len(grid),
        "n_beat_cap_disc": int(len(both)),
        "best_stable": stable,
        "best_cap_and_disc": best_both,
        "best_overall": best_any,
        "prior_default": rows[2] if len(rows) > 2 else None,
        "promote": promote,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({"promote": promote, "pick": pick}, indent=2, default=str))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
