#!/usr/bin/env python3
"""Ablate entry filters to lift per-trade win rate on stock_flow_opt Jul pocket.

Baseline frozen: stk_d0.003_f120_sh0.6_tp0.25_sl0.2
  n=164 win≈47% day_win=70% add≈+33% (research_stock_flow_opt_jul10_23)

Does NOT retune tp/sl grid on Jul. Only adds causal stock filters.
Dual-window / quote left for a later gate.

Example:
  PYTHONPATH=. python -m maga7.tools.run_stock_flow_winrate_ablation \\
    --tag research_stock_flow_winrate_ablation_jul10_23
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

from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.config import load_profile
from maga7.common.open_lock import (
    load_multidte_lock_index,
    resolve_open_lock_contract,
    resolve_otm_rungs,
)
from maga7.common.option_flow import DEFAULT_TICK_ROOT, load_option_tick_day, tick_dates
from maga7.common.option_trade_tpsl import simulate_trade_tpsl
from maga7.common.replay import to_ny
from maga7.common.smc_flow import dn_vol_share_at, prepare_smc_flow_day
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_certainty_morph_tpsl import _stats
from maga7.tools.scan_session_horizon_foresight import (
    _paths_by_ticker,
    _spot_at_arr,
    _stock_arrays,
)

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

# Frozen champion entry (exit fixed)
BASE_DISP = 0.003
BASE_FLOW = 120
BASE_SHARE = 0.60
BASE_TP = 0.25
BASE_SL = 0.20
BASE_DISP_SEC = 60


def _idx_at_or_before(ts_ns: np.ndarray, t_ns: int) -> int | None:
    i = int(np.searchsorted(ts_ns, t_ns, side="right") - 1)
    return i if i >= 0 else None


def _feat_at(
    arrays: dict[str, Any], *, t: pd.Timestamp, flow_sec: int, disp_sec: int
) -> dict[str, float] | None:
    ts_ns = arrays["ts_ns"]
    c = arrays["close"]
    i = _idx_at_or_before(ts_ns, int(to_ny(t).value))
    if i is None or i < max(flow_sec, disp_sec, 30):
        return None
    share = dn_vol_share_at(arrays, i=i, window_sec=int(flow_sec))
    if share is None:
        return None
    j0 = _idx_at_or_before(ts_ns, int(ts_ns[i]) - int(disp_sec) * 1_000_000_000)
    if j0 is None or j0 >= i:
        return None
    a, b = float(c[j0]), float(c[i])
    if a <= 0 or not np.isfinite(a) or not np.isfinite(b):
        return None
    mf = float(arrays["mf"][i]) if np.isfinite(arrays["mf"][i]) else float("nan")
    sd = int(arrays["streak_dn"][i])
    vz = float(arrays["vol_z"][i]) if "vol_z" in arrays and np.isfinite(arrays["vol_z"][i]) else float("nan")
    return {
        "dn_vol_share": float(share),
        "stock_ret_disp": float(b / a - 1.0),
        "mf": mf,
        "streak_dn": float(sd),
        "vol_z": vz,
        "i": float(i),
    }


FilterFn = Callable[[dict[str, float]], bool]


def _variants() -> list[tuple[str, FilterFn, dict[str, Any]]]:
    """name, extra_filter(feat)->bool, gate overrides."""

    def always(_f: dict[str, float]) -> bool:
        return True

    return [
        ("baseline", always, {}),
        ("share65", always, {"min_dn_share": 0.65}),
        ("disp005", always, {"disp_thr": 0.005}),
        ("disp005_share65", always, {"disp_thr": 0.005, "min_dn_share": 0.65}),
        ("mf_neg", lambda f: np.isfinite(f["mf"]) and f["mf"] < 0, {}),
        ("streak3", lambda f: f["streak_dn"] >= 3, {}),
        ("streak5", lambda f: f["streak_dn"] >= 5, {}),
        ("mf_neg_streak3", lambda f: np.isfinite(f["mf"]) and f["mf"] < 0 and f["streak_dn"] >= 3, {}),
        ("volz15", lambda f: np.isfinite(f["vol_z"]) and f["vol_z"] >= 1.5, {}),
        ("volz20", lambda f: np.isfinite(f["vol_z"]) and f["vol_z"] >= 2.0, {}),
        (
            "mf_neg_share65",
            lambda f: np.isfinite(f["mf"]) and f["mf"] < 0,
            {"min_dn_share": 0.65},
        ),
        (
            "disp005_mf_neg_streak3",
            lambda f: np.isfinite(f["mf"]) and f["mf"] < 0 and f["streak_dn"] >= 3,
            {"disp_thr": 0.005},
        ),
        (
            "quality_pack",
            lambda f: (
                np.isfinite(f["mf"])
                and f["mf"] < 0
                and f["streak_dn"] >= 3
                and (not np.isfinite(f["vol_z"]) or f["vol_z"] >= 1.5)
            ),
            {"min_dn_share": 0.65},
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_stock_flow_winrate_ablation_jul10_23")
    ap.add_argument("--tick-root", default=str(DEFAULT_TICK_ROOT))
    ap.add_argument("--start-date", default="2026-07-10")
    ap.add_argument("--end-date", default="2026-07-23")
    ap.add_argument("--stride-sec", type=int, default=5)
    ap.add_argument("--rearm-gap-sec", type=int, default=60)
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=4)
    ap.add_argument("--cooldown-minutes", type=float, default=1.0)
    ap.add_argument("--target-win", type=float, default=0.55)
    args = ap.parse_args(argv)

    tick_root = Path(args.tick_root)
    dates = tick_dates(tick_root)
    if args.start_date:
        dates = [d for d in dates if d >= args.start_date]
    if args.end_date:
        dates = [d for d in dates if d <= args.end_date]
    if not dates:
        print("no dates in range", flush=True)
        return 2
    variants = _variants()

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    symbols = list(prof.get("symbols") or [])
    stock_1s = Path(paths.get("stock_1s_root") or "/mnt/s990/data/raw_1s/stocks").expanduser()
    lock = load_multidte_lock_index(Path(paths["open_locked_map"]).expanduser())
    otm = resolve_otm_rungs(prof, default=3)
    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    print(
        f"winrate ablation {dates[0]}..{dates[-1]} variants={len(variants)} "
        f"base=d{BASE_DISP}_f{BASE_FLOW}_sh{BASE_SHARE}_tp{BASE_TP}_sl{BASE_SL}",
        flush=True,
    )

    # Collect arms per variant
    arms_by: dict[str, list[dict[str, Any]]] = {n: [] for n, _, _ in variants}

    for di, date in enumerate(dates):
        if di % 2 == 0:
            n0 = len(arms_by["baseline"])
            print(f"[day] {date} ({di+1}/{len(dates)}) baseline_arms={n0}", flush=True)
        for sym in symbols:
            day = load_stock_1s_day(stock_1s, sym, date)
            arrays = prepare_smc_flow_day(day)
            if arrays is None:
                continue
            # attach vol_z already in prepare_smc via sec_mf
            if "vol_z" not in arrays and "vol_z" in (day.columns if day is not None else []):
                pass
            # ensure vol_z on arrays from prepare — sec_mf has vol_z on dataframe;
            # prepare_smc_flow_day stores mf/streak but check for vol_z
            ts_ns, px = _stock_arrays(day)
            tday = load_option_tick_day(tick_root, sym, date)
            if tday is None or tday.empty:
                continue
            tpaths = _paths_by_ticker(tday)
            by_dte = lock.get((sym, date))
            if not by_dte or not tpaths:
                continue

            # vol_z: rebuild from prepare output if missing
            if "vol_z" not in arrays:
                from maga7.common.sec_mf import attach_sec_mf_features

                feat_df = attach_sec_mf_features(day)
                if feat_df is not None and not feat_df.empty and "vol_z" in feat_df.columns:
                    # align by timestamp
                    arrays = dict(arrays)
                    # prepare_smc already sorted; re-pull vol_z same length
                    vz = pd.to_numeric(feat_df["vol_z"], errors="coerce").to_numpy(dtype=np.float64)
                    if len(vz) == int(arrays["n"]):
                        arrays["vol_z"] = vz

            for sess_name, s0, s1 in SESSIONS:
                warm = max(BASE_FLOW, BASE_DISP_SEC, 120)
                t_start = pd.Timestamp(f"{date} {s0}:00", tz=NY) + pd.Timedelta(seconds=warm)
                t_end = pd.Timestamp(f"{date} {s1}:00", tz=NY)
                for vname, filt, over in variants:
                    disp_thr = float(over.get("disp_thr", BASE_DISP))
                    flow_sec = int(over.get("flow_sec", BASE_FLOW))
                    min_share = float(over.get("min_dn_share", BASE_SHARE))
                    t = t_start
                    next_ok = t
                    prev_on = False
                    stride = pd.Timedelta(seconds=int(args.stride_sec))
                    gap = pd.Timedelta(seconds=int(args.rearm_gap_sec))
                    while t < t_end:
                        feat = _feat_at(
                            arrays, t=t, flow_sec=flow_sec, disp_sec=BASE_DISP_SEC
                        )
                        on = bool(
                            feat is not None
                            and feat["stock_ret_disp"] <= -disp_thr
                            and feat["dn_vol_share"] >= min_share
                            and filt(feat)
                        )
                        fire = on and (t >= next_ok) and (not prev_on)
                        if fire and feat is not None:
                            spot = _spot_at_arr(ts_ns, px, t)
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
                            if ticker:
                                arr = tpaths.get(str(ticker).replace("O:", ""))
                                if arr is not None:
                                    arms_by[vname].append(
                                        {
                                            "date": date,
                                            "symbol": sym,
                                            "session": sess_name,
                                            "arm_ts": to_ny(t),
                                            "ticker": ticker,
                                            "dte": dte,
                                            "pts": arr[0],
                                            "plast": arr[1],
                                            "variant": vname,
                                            **{k: feat[k] for k in feat if k != "i"},
                                        }
                                    )
                                    next_ok = t + gap
                        prev_on = on
                        t += stride

    rows: list[dict[str, Any]] = []
    trade_dump: dict[str, pd.DataFrame] = {}
    base_st: dict[str, Any] | None = None

    for vname, _, over in variants:
        raw: list[dict] = []
        for arm in arms_by[vname]:
            sim = simulate_trade_tpsl(
                arm["pts"],
                arm["plast"],
                arm["arm_ts"],
                tp=BASE_TP,
                sl=BASE_SL,
                max_hold_sec=int(args.max_hold_sec),
                slip=float(args.slip),
            )
            if sim is None or not np.isfinite(sim["ret"]):
                continue
            et = to_ny(arm["arm_ts"])
            raw.append(
                {
                    "date": arm["date"],
                    "symbol": arm["symbol"],
                    "dir": "DN",
                    "session": arm["session"],
                    "entry_ts": str(et),
                    "exit_ts": str(et + pd.Timedelta(seconds=sim["hold_sec"])),
                    "ticker": arm["ticker"],
                    "ret": sim["ret"],
                    "exit_reason": sim["reason"],
                    "hold_sec": sim["hold_sec"],
                    "cell": vname,
                    "event_source": "stock_flow_winrate",
                }
            )
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
        if vname == "baseline":
            base_st = st
        row = {
            "variant": vname,
            "overrides": json.dumps(over),
            "n_arms": len(arms_by[vname]),
            **st,
        }
        if base_st and st.get("n"):
            row["d_win"] = float(st["win"] or 0) - float(base_st.get("win") or 0)
            row["d_add"] = float(st["add"] or 0) - float(base_st.get("add") or 0)
            row["d_mean"] = float(st["mean"] or 0) - float(base_st.get("mean") or 0)
            row["hit_target_win"] = bool(float(st.get("win") or 0) >= float(args.target_win))
            row["keep_edge"] = bool(
                float(st.get("mean") or 0) > 0
                and float(st.get("add") or 0) > 0
                and float(st.get("day_win") or 0) >= 0.55
            )
        rows.append(row)
        if sized:
            trade_dump[vname] = pd.DataFrame(sized)
        print(
            f"  {vname}: n={st.get('n')} win={st.get('win')} mean={st.get('mean')} "
            f"day_win={st.get('day_win')} add={st.get('add')}",
            flush=True,
        )

    score = pd.DataFrame(rows)
    score.to_csv(out / "scoreboard.csv", index=False)
    for name in ("baseline",) + tuple(
        score.sort_values("win", ascending=False)["variant"].head(5)
    ):
        if name in trade_dump:
            trade_dump[name].to_csv(out / f"trades_{name}.csv", index=False)

    # pick: max win among keep_edge; else max win with mean>0
    cand = score.copy()
    if "keep_edge" in cand.columns:
        ok = cand[cand.keep_edge & cand.hit_target_win]
        if ok.empty:
            ok = cand[cand.keep_edge]
        if ok.empty:
            ok = cand[(cand["mean"].fillna(0) > 0) & (cand["add"].fillna(0) > 0)]
    else:
        ok = cand
    best = ok.sort_values(["win", "add"], ascending=[False, False]).iloc[0].to_dict() if len(ok) else None

    summary = {
        "baseline_frozen": {
            "cell": f"stk_d{BASE_DISP}_f{BASE_FLOW}_sh{BASE_SHARE}_tp{BASE_TP}_sl{BASE_SL}",
            "stats": base_st,
        },
        "target_win": args.target_win,
        "best_keep_edge": best,
        "n_hit_target_and_edge": int(
            ((score.get("hit_target_win") == True) & (score.get("keep_edge") == True)).sum()
        )
        if "hit_target_win" in score.columns
        else 0,
        "verdict": (
            "WINRATE_LIFT"
            if best and float(best.get("win") or 0) >= float(args.target_win)
            else "WINRATE_PARTIAL"
            if best and float(best.get("d_win") or 0) > 0.02
            else "WINRATE_NO_LIFT"
        ),
        "note": "Jul pocket only; dual-window/quote deferred.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n=== verdict", summary["verdict"], flush=True)
    cols = [
        c
        for c in [
            "variant",
            "n",
            "win",
            "mean",
            "day_win",
            "add",
            "d_win",
            "d_add",
            "hit_target_win",
            "keep_edge",
        ]
        if c in score.columns
    ]
    print(score.sort_values("win", ascending=False)[cols].to_string(index=False), flush=True)
    if best:
        print(
            f"\nbest {best.get('variant')}: win={best.get('win')} n={best.get('n')} "
            f"add={best.get('add')} d_win={best.get('d_win')} d_add={best.get('d_add')}",
            flush=True,
        )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
