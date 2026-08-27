#!/usr/bin/env python3
"""Try stock-activity variants for AM MF timing — one-by-one protocols.

① stock_act: stock vol_z / volume_ratio fires → MF/ret picks call/put → TP/SL
② pocket_gate: existing no_b_up multi-gate entries + activity tighten + exit grid

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_stock_activity_variants \\
    --tag research_am_stock_activity_variants --protocol both
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
from maga7.common.open_lock import (
    load_multidte_lock_index,
    resolve_open_lock_contract,
    resolve_otm_rungs,
)
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import to_ny
from maga7.common.session_1s_features import features_at, prepare_day_arrays
from maga7.common.stock_1s import session_dates
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_activity_mf_scalp import _dir_from_stock
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


def _score_row(
    raw: list[dict[str, Any]],
    *,
    position_frac: float,
    max_concurrent: int,
    cooldown: float,
) -> dict[str, Any]:
    disc = [t for t in raw if t["calendar"] == "may_jul09"]
    blind = [t for t in raw if t["calendar"] == "jul10_23"]
    sized_d = _portfolio_day(
        sorted(disc, key=lambda x: (x["entry_ts"], x["symbol"])),
        position_frac=position_frac,
        max_concurrent=max_concurrent,
        cooldown_minutes=cooldown,
    )
    sized_b = _portfolio_day(
        sorted(blind, key=lambda x: (x["entry_ts"], x["symbol"])),
        position_frac=position_frac,
        max_concurrent=max_concurrent,
        cooldown_minutes=cooldown,
    )
    st_d = _equity_stats(pd.DataFrame(sized_d))
    st_b = _equity_stats(pd.DataFrame(sized_b))
    months = _month_compounds(pd.DataFrame(sized_d + sized_b))
    row: dict[str, Any] = {
        "n_raw": len(raw),
        "frac_up": float(np.mean([t["dir"] == "UP" for t in raw])) if raw else 0.0,
        "may": months.get("2026-05"),
        "jun": months.get("2026-06"),
        "jul": months.get("2026-07"),
    }
    for k, v in st_d.items():
        row[f"disc_{k}"] = v
    for k, v in st_b.items():
        row[f"blind_{k}"] = v
    return row


def _soft_pass(row: dict[str, Any]) -> bool:
    return bool(
        int(row.get("disc_n") or 0) >= 15
        and float(row.get("disc_trade_win") or 0) >= 0.55
        and float(row.get("disc_maxdd") or -9) >= -0.22
        and float(row.get("disc_compound") or -9) > 0
        and float(row.get("may") or -9) > 0
        and int(row.get("blind_n") or 0) >= 4
        and float(row.get("blind_compound") or -9) > 0
    )


def run_stock_act(
    *,
    prof: dict[str, Any],
    trades_root: Path,
    out: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """① Stock vol activity → MF direction → exits incl h240."""
    stock_1s = Path(prof["_paths"]["stock_1s_root"])
    lock = load_multidte_lock_index(Path(prof["_paths"]["open_locked_map"]).expanduser())
    otm = resolve_otm_rungs(prof, default=3)
    symbols = list(prof.get("symbols") or [])
    start_all = min(w[1] for w in WINDOWS)
    end_all = max(w[2] for w in WINDOWS)
    dates = [d for d in session_dates(start_all, end_all) if start_all <= d <= end_all]

    # activity specs × dir × exit
    act_specs = [
        ("volz15", lambda f: np.isfinite(f.get("vol_z")) and float(f["vol_z"]) >= 1.5),
        ("volz20", lambda f: np.isfinite(f.get("vol_z")) and float(f["vol_z"]) >= 2.0),
        ("volz25", lambda f: np.isfinite(f.get("vol_z")) and float(f["vol_z"]) >= 2.5),
        ("volr12", lambda f: np.isfinite(f.get("volume_ratio_60")) and float(f["volume_ratio_60"]) >= 1.2),
        ("volr15", lambda f: np.isfinite(f.get("volume_ratio_60")) and float(f["volume_ratio_60"]) >= 1.5),
        ("volz15+volr12", lambda f: (
            np.isfinite(f.get("vol_z")) and float(f["vol_z"]) >= 1.5
            and np.isfinite(f.get("volume_ratio_60")) and float(f["volume_ratio_60"]) >= 1.2
        )),
    ]
    dir_modes = ("mf100+ret60", "mf100+ret60+volr12", "mf100")
    exits = (
        (0.08, 0.10, 30),
        (0.08, 0.15, 60),
        (0.12, 0.10, 45),
        (0.08, 0.15, 240),
        (0.10, 0.15, 240),
    )

    path_cache: dict[tuple[str, str], dict] = {}

    def pack(date: str, sym: str):
        key = (date, sym)
        if key not in path_cache:
            tday = load_option_trades(trades_root, sym, date)
            path_cache[key] = {
                "by_ticker": _paths_by_ticker(tday) if tday is not None and not tday.empty else {},
            }
        return path_cache[key]

    # family = (act_name, dir_mode) → arms
    families: dict[tuple[str, str], list[dict[str, Any]]] = {
        (a, d): [] for a, _ in act_specs for d in dir_modes
    }

    print(f"[① stock_act] days={len(dates)} families={len(families)}", flush=True)
    for di, date in enumerate(dates):
        if di % 15 == 0:
            print(f"  day {date} ({di+1}/{len(dates)})", flush=True)
        cal = _window_of(date)
        if cal is None:
            continue
        for sym in symbols:
            raw = load_stock_1s_day(stock_1s, sym, date)
            if raw is None or raw.empty:
                continue
            sarr = prepare_day_arrays(raw)
            by_dte = lock.get((sym, date))
            if not by_dte:
                continue
            t0 = to_ny(pd.Timestamp(f"{date} {args.window_start}", tz=NY))
            t1 = to_ny(pd.Timestamp(f"{date} {args.window_end}", tz=NY))
            ts_ns = sarr["ts_ns"]
            i0 = int(np.searchsorted(ts_ns, int(t0.value), side="left"))
            i1 = int(np.searchsorted(ts_ns, int(t1.value), side="right") - 1)
            if i1 <= i0:
                continue
            opt = pack(date, sym)
            last_fire = {k: -10**18 for k in families}
            n_fire = {k: 0 for k in families}
            stride = max(1, int(args.stride_sec))
            for i in range(i0, i1 + 1, stride):
                t = pd.Timestamp(int(ts_ns[i]), tz="UTC").tz_convert(NY)
                feat = features_at(sarr, t)
                if feat is None:
                    continue
                spot = float(feat.get("px") or np.nan)
                if not np.isfinite(spot):
                    continue
                for act_name, act_fn in act_specs:
                    if not act_fn(feat):
                        continue
                    for dmode in dir_modes:
                        key = (act_name, dmode)
                        if n_fire[key] >= int(args.max_arms_per_sym_day):
                            continue
                        if (int(ts_ns[i]) - last_fire[key]) / 1e9 < float(args.rearm_gap_sec):
                            continue
                        direction = _dir_from_stock(feat, dmode)
                        if direction is None:
                            continue
                        ticker, dte, _ = resolve_open_lock_contract(
                            by_dte,
                            direction=direction,
                            moneyness="ATM",
                            spot=spot,
                            prefer_dte=0,
                            allowed_dte=(0, 1, 2),
                            clear_otm_thresh=0.01,
                            ladder=True,
                            otm_rungs=otm,
                        )
                        if not ticker:
                            continue
                        path = opt["by_ticker"].get(str(ticker).replace("O:", ""))
                        if path is None:
                            continue
                        families[key].append(
                            {
                                "date": date,
                                "symbol": sym,
                                "dir": direction,
                                "entry_ts": t,
                                "calendar": cal,
                                "path": path,
                                "vol_z": float(feat.get("vol_z") or np.nan),
                                "volume_ratio_60": float(feat.get("volume_ratio_60") or np.nan),
                            }
                        )
                        last_fire[key] = int(ts_ns[i])
                        n_fire[key] += 1

    score_rows: list[dict[str, Any]] = []
    for (act_name, dmode), arms in families.items():
        prepared = []
        for a in arms:
            win = _path_window(
                a["path"][0], a["path"][1], a["entry_ts"], max_hold_sec=300, slip=float(args.slip)
            )
            if win is None:
                continue
            prepared.append({**a, "rets": win[0], "holds": win[1]})
        print(f"  family {act_name}/{dmode}: arms={len(arms)} prepared={len(prepared)}", flush=True)
        for tp, sl, h in exits:
            raw = []
            for p in prepared:
                sim = simulate_exit(
                    p["rets"], p["holds"], mode="tpsl",
                    params={"tp": tp, "sl": sl, "max_hold": h},
                )
                if not np.isfinite(sim.get("ret", np.nan)):
                    continue
                et = p["entry_ts"]
                raw.append(
                    {
                        "date": p["date"],
                        "symbol": p["symbol"],
                        "dir": p["dir"],
                        "calendar": p["calendar"],
                        "entry_ts": et,
                        "exit_ts": et + pd.Timedelta(seconds=float(sim["hold_sec"])),
                        "ret": float(sim["ret"]),
                        "exit_reason": str(sim["reason"]),
                        "hold_sec": float(sim["hold_sec"]),
                    }
                )
            row = _score_row(
                raw,
                position_frac=float(args.position_frac),
                max_concurrent=int(args.max_concurrent),
                cooldown=float(args.cooldown_minutes),
            )
            row.update(
                {
                    "protocol": "stock_act",
                    "policy": f"stk_{act_name}_{dmode}_tp{tp:g}_sl{sl:g}_h{h}",
                    "act": act_name,
                    "dir_mode": dmode,
                    "tp": tp,
                    "sl": sl,
                    "max_hold": h,
                    "soft_pass": _soft_pass(row),
                }
            )
            score_rows.append(row)

    sb = pd.DataFrame(score_rows)
    sb.to_csv(out / "scoreboard_stock_act.csv", index=False)
    return sb


def run_pocket_gate(
    *,
    prof: dict[str, Any],
    trades_root: Path,
    enriched_path: Path,
    out: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """② Existing pocket multi-gate + activity tighten + exit grid."""
    probes = pd.read_csv(enriched_path)
    probes["entry_ts"] = pd.to_datetime(probes["entry_ts"])
    probes = probes[probes["enrich_ok"] == True].copy()  # noqa: E712
    pdf = pd.DataFrame(sorted(POCKET_SETS["no_b_up"]), columns=["session", "tod_bucket", "dir"])
    probes = probes.merge(pdf, on=["session", "tod_bucket", "dir"], how="inner")
    gates = dict(build_gates())
    entry_names = ("vd_soft", "vd+cont60+mf100+volr12", "vd+volr12")
    act_filters = [
        ("none", lambda r: True),
        ("volz15", lambda r: float(r.get("vol_z") or np.nan) >= 1.5),
        ("volz20", lambda r: float(r.get("vol_z") or np.nan) >= 2.0),
        ("volr12", lambda r: float(r.get("volume_ratio_60") or np.nan) >= 1.2),
        ("volr15", lambda r: float(r.get("volume_ratio_60") or np.nan) >= 1.5),
        ("volz15+volr12", lambda r: (
            float(r.get("vol_z") or np.nan) >= 1.5
            and float(r.get("volume_ratio_60") or np.nan) >= 1.2
        )),
    ]
    exits = (
        (0.08, 0.15, 240),
        (0.08, 0.10, 60),
        (0.12, 0.10, 45),
        (0.08, 0.10, 30),
        (0.10, 0.15, 240),
    )

    path_cache: dict[tuple[str, str], dict] = {}

    def paths(date: str, sym: str):
        key = (date, sym)
        if key not in path_cache:
            tday = load_option_trades(trades_root, sym, date)
            path_cache[key] = _paths_by_ticker(tday) if tday is not None and not tday.empty else {}
        return path_cache[key]

    # prepare all pocket probes once
    base_prep: list[dict[str, Any]] = []
    for _, r in probes.iterrows():
        date, sym = str(r["date"]), str(r["symbol"])
        if _window_of(date) is None:
            continue
        arrs = paths(date, sym)
        arr = arrs.get(str(r["ticker"]).replace("O:", ""))
        if arr is None:
            continue
        et = to_ny(pd.Timestamp(r["entry_ts"]))
        win = _path_window(arr[0], arr[1], et, max_hold_sec=300, slip=float(args.slip))
        if win is None:
            continue
        base_prep.append(
            {
                "row": r,
                "date": date,
                "symbol": sym,
                "dir": str(r["dir"]),
                "calendar": str(r["calendar"]),
                "entry_ts": et,
                "rets": win[0],
                "holds": win[1],
            }
        )
    print(f"[② pocket_gate] prepared={len(base_prep)}", flush=True)

    score_rows: list[dict[str, Any]] = []
    for ename in entry_names:
        gfn = gates[ename]
        for aname, afn in act_filters:
            subset = [p for p in base_prep if gfn(p["row"]) and afn(p["row"])]
            for tp, sl, h in exits:
                raw = []
                for p in subset:
                    sim = simulate_exit(
                        p["rets"], p["holds"], mode="tpsl",
                        params={"tp": tp, "sl": sl, "max_hold": h},
                    )
                    if not np.isfinite(sim.get("ret", np.nan)):
                        continue
                    et = p["entry_ts"]
                    raw.append(
                        {
                            "date": p["date"],
                            "symbol": p["symbol"],
                            "dir": p["dir"],
                            "calendar": p["calendar"],
                            "entry_ts": et,
                            "exit_ts": et + pd.Timedelta(seconds=float(sim["hold_sec"])),
                            "ret": float(sim["ret"]),
                            "exit_reason": str(sim["reason"]),
                            "hold_sec": float(sim["hold_sec"]),
                        }
                    )
                row = _score_row(
                    raw,
                    position_frac=float(args.position_frac),
                    max_concurrent=int(args.max_concurrent),
                    cooldown=float(args.cooldown_minutes),
                )
                row.update(
                    {
                        "protocol": "pocket_gate",
                        "policy": f"{ename}|act_{aname}|tp{tp:g}_sl{sl:g}_h{h}",
                        "entry": ename,
                        "act": aname,
                        "tp": tp,
                        "sl": sl,
                        "max_hold": h,
                        "soft_pass": _soft_pass(row),
                    }
                )
                score_rows.append(row)
                if aname == "none" and h == 240 and tp == 0.08:
                    print(
                        f"  baseline {ename}: n={row['n_raw']} "
                        f"cmp={row.get('disc_compound')} blind={row.get('blind_compound')}",
                        flush=True,
                    )

    sb = pd.DataFrame(score_rows)
    sb.to_csv(out / "scoreboard_pocket_gate.csv", index=False)
    return sb


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--enriched", default=str(DEFAULT_ENRICHED))
    ap.add_argument("--tag", default="research_am_stock_activity_variants")
    ap.add_argument("--protocol", choices=("stock_act", "pocket_gate", "both"), default="both")
    ap.add_argument("--window-start", default="09:30")
    ap.add_argument("--window-end", default="11:30")
    ap.add_argument("--stride-sec", type=int, default=5)
    ap.add_argument("--rearm-gap-sec", type=int, default=60)
    ap.add_argument("--max-arms-per-sym-day", type=int, default=3)
    ap.add_argument("--position-frac", type=float, default=0.20)
    ap.add_argument("--max-concurrent", type=int, default=5)
    ap.add_argument("--cooldown-minutes", type=float, default=5.0)
    ap.add_argument("--slip", type=float, default=0.01)
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    out = Path(prof["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    trades_root = Path(args.trades_root)

    frames: list[pd.DataFrame] = []
    if args.protocol in ("stock_act", "both"):
        frames.append(run_stock_act(prof=prof, trades_root=trades_root, out=out, args=args))
    if args.protocol in ("pocket_gate", "both"):
        frames.append(
            run_pocket_gate(
                prof=prof,
                trades_root=trades_root,
                enriched_path=Path(args.enriched),
                out=out,
                args=args,
            )
        )

    sb = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    sb.to_csv(out / "scoreboard.csv", index=False)

    cols = [
        c
        for c in [
            "protocol",
            "policy",
            "n_raw",
            "disc_n",
            "disc_trade_win",
            "disc_maxdd",
            "disc_compound",
            "blind_n",
            "blind_trade_win",
            "blind_compound",
            "may",
            "jun",
            "jul",
            "soft_pass",
        ]
        if c in sb.columns
    ]
    verdict: dict[str, Any] = {"tag": args.tag, "protocols": {}}
    for proto, g in sb.groupby("protocol"):
        soft = g[g["soft_pass"] == True].sort_values(  # noqa: E712
            "disc_compound", ascending=False
        )
        top = g.sort_values("disc_compound", ascending=False).head(8)
        verdict["protocols"][str(proto)] = {
            "n": int(len(g)),
            "soft_pass_n": int(len(soft)),
            "top_soft": soft.head(10).to_dict(orient="records") if len(soft) else [],
            "top_disc": top.to_dict(orient="records"),
        }
        print(f"\n=== {proto} soft_pass={len(soft)}/{len(g)} ===", flush=True)
        print(soft[cols].head(8).to_string(index=False) if len(soft) else "(none)", flush=True)
        print(f"\n{proto} top disc (any):", flush=True)
        print(top[cols].head(6).to_string(index=False), flush=True)

    (out / "summary.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
