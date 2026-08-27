#!/usr/bin/env python3
"""Lift stock_flow_opt trade-win via SMC/ICT structure + session VWAP filters.

Frozen base gate (Jul pocket):
  ret_60s <= -0.3%, dn_vol_share_120 >= 0.60, rising fire, tp25/sl20.

Adds causal filters only (no tp/sl retune):
  - below session VWAP / deep below VWAP (ICT discount)
  - BOS through prior swing low / sweep-reject-dn
  - combinations with vol_z>=1.5

Example:
  PYTHONPATH=. python -m maga7.tools.run_stock_flow_smc_vwap_ablation \\
    --tag research_stock_flow_smc_vwap_jul10_23
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
from maga7.common.option_flow import DEFAULT_TICK_ROOT, load_option_tick_day, tick_dates
from maga7.common.option_trade_tpsl import simulate_trade_tpsl
from maga7.common.replay import to_ny
from maga7.common.smc_flow import detect_smc_flow_dn, dn_vol_share_at, prepare_smc_flow_day
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

BASE_DISP = 0.003
BASE_FLOW = 120
BASE_SHARE = 0.60
BASE_TP = 0.25
BASE_SL = 0.20
BASE_DISP_SEC = 60
SWING_SEC = 300


def _idx_at_or_before(ts_ns: np.ndarray, t_ns: int) -> int | None:
    i = int(np.searchsorted(ts_ns, t_ns, side="right") - 1)
    return i if i >= 0 else None


def _session_vwap_arrays(arrays: dict[str, Any]) -> np.ndarray:
    """Causal session VWAP from day open of arrays (full RTH slice)."""
    c = arrays["close"]
    v = arrays["volume"]
    pv = np.cumsum(c * v)
    vv = np.cumsum(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = pv / vv
    out[vv <= 0] = np.nan
    return out


def _base_feat(arrays: dict[str, Any], *, i: int) -> dict[str, float] | None:
    ts_ns = arrays["ts_ns"]
    c = arrays["close"]
    if i < max(BASE_FLOW, BASE_DISP_SEC, 30):
        return None
    share = dn_vol_share_at(arrays, i=i, window_sec=BASE_FLOW)
    if share is None:
        return None
    j0 = _idx_at_or_before(ts_ns, int(ts_ns[i]) - BASE_DISP_SEC * 1_000_000_000)
    if j0 is None or j0 >= i:
        return None
    a, b = float(c[j0]), float(c[i])
    if a <= 0 or not np.isfinite(a) or not np.isfinite(b):
        return None
    ret = b / a - 1.0
    if ret > -BASE_DISP or share < BASE_SHARE:
        return None
    mf = float(arrays["mf"][i]) if np.isfinite(arrays["mf"][i]) else float("nan")
    sd = float(arrays["streak_dn"][i])
    if "vol_z" in arrays and np.isfinite(arrays["vol_z"][i]):
        vz = float(arrays["vol_z"][i])
    else:
        vz = float("nan")
    return {
        "dn_vol_share": float(share),
        "stock_ret_disp": float(ret),
        "mf": mf,
        "streak_dn": sd,
        "vol_z": vz,
    }


def _variants() -> list[tuple[str, dict[str, Any]]]:
    # flags consumed in fire loop
    return [
        ("baseline", {}),
        ("volz15", {"volz": 1.5}),
        ("below_vwap", {"below_vwap": True}),
        ("vwap_deep_10bps", {"below_vwap": True, "min_vwap_gap": 0.001}),
        ("vwap_deep_20bps", {"below_vwap": True, "min_vwap_gap": 0.002}),
        ("bos", {"morph": "bos_disp_dn"}),
        ("sweep", {"morph": "sweep_rev_dn"}),
        ("bos_below_vwap", {"morph": "bos_disp_dn", "below_vwap": True}),
        ("bos_vwap_deep", {"morph": "bos_disp_dn", "below_vwap": True, "min_vwap_gap": 0.001}),
        ("bos_volz15", {"morph": "bos_disp_dn", "volz": 1.5}),
        ("below_vwap_volz15", {"below_vwap": True, "volz": 1.5}),
        ("bos_below_vwap_volz15", {"morph": "bos_disp_dn", "below_vwap": True, "volz": 1.5}),
        ("sweep_below_vwap", {"morph": "sweep_rev_dn", "below_vwap": True}),
        ("ict_pack", {"morph": "bos_disp_dn", "below_vwap": True, "min_vwap_gap": 0.001, "volz": 1.5}),
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_stock_flow_smc_vwap_jul10_23")
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

    dates = tick_dates(args.tick_root)
    dates = [d for d in dates if args.start_date <= d <= args.end_date]
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
        f"SMC/VWAP winrate {dates[0]}..{dates[-1]} variants={len(variants)}",
        flush=True,
    )
    arms_by: dict[str, list[dict]] = {n: [] for n, _ in variants}

    for di, date in enumerate(dates):
        if di % 2 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)}) base={len(arms_by['baseline'])}", flush=True)
        for sym in symbols:
            day = load_stock_1s_day(stock_1s, sym, date)
            arrays = prepare_smc_flow_day(day)
            if arrays is None:
                continue
            vwap = _session_vwap_arrays(arrays)
            ts_ns, px = _stock_arrays(day)
            tday = load_option_tick_day(args.tick_root, sym, date)
            if tday is None or tday.empty:
                continue
            tpaths = _paths_by_ticker(tday)
            by_dte = lock.get((sym, date))
            if not by_dte or not tpaths:
                continue
            for sess_name, s0, s1 in SESSIONS:
                warm = max(BASE_FLOW, BASE_DISP_SEC, SWING_SEC, 120)
                t_start = pd.Timestamp(f"{date} {s0}:00", tz=NY) + pd.Timedelta(seconds=warm)
                t_end = pd.Timestamp(f"{date} {s1}:00", tz=NY)
                for vname, cfg in variants:
                    t = t_start
                    next_ok = t
                    prev_on = False
                    stride = pd.Timedelta(seconds=int(args.stride_sec))
                    gap = pd.Timedelta(seconds=int(args.rearm_gap_sec))
                    while t < t_end:
                        i = _idx_at_or_before(arrays["ts_ns"], int(to_ny(t).value))
                        feat = _base_feat(arrays, i=i) if i is not None else None
                        on = feat is not None
                        if on and i is not None:
                            px_i = float(arrays["close"][i])
                            vw = float(vwap[i]) if np.isfinite(vwap[i]) else float("nan")
                            gap_vw = (vw - px_i) / vw if np.isfinite(vw) and vw > 0 else float("nan")
                            if cfg.get("below_vwap"):
                                if not (np.isfinite(gap_vw) and gap_vw > 0):
                                    on = False
                            min_gap = cfg.get("min_vwap_gap")
                            if on and min_gap is not None:
                                if not (np.isfinite(gap_vw) and gap_vw >= float(min_gap)):
                                    on = False
                            vz_need = cfg.get("volz")
                            if on and vz_need is not None:
                                if not (
                                    np.isfinite(feat["vol_z"])
                                    and feat["vol_z"] >= float(vz_need)
                                ):
                                    on = False
                            morph = cfg.get("morph")
                            if on and morph:
                                arm = detect_smc_flow_dn(
                                    arrays,
                                    i=i,
                                    morph=str(morph),
                                    swing_sec=SWING_SEC,
                                    disp_sec=BASE_DISP_SEC,
                                    disp_thr=BASE_DISP,
                                    flow_sec=BASE_FLOW,
                                    min_dn_vol_share=BASE_SHARE,
                                    min_streak_dn=0,
                                    require_mf_neg=False,
                                )
                                if arm is None:
                                    on = False
                        fire = on and (t >= next_ok) and (not prev_on)
                        if fire and feat is not None and i is not None:
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
                                            **feat,
                                        }
                                    )
                                    next_ok = t + gap
                        prev_on = bool(on)
                        t += stride

    rows: list[dict[str, Any]] = []
    trade_dump: dict[str, pd.DataFrame] = {}
    base_st: dict[str, Any] | None = None
    for vname, cfg in variants:
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
                    "event_source": "stock_flow_smc_vwap",
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
        row = {"variant": vname, "cfg": json.dumps(cfg), "n_arms": len(arms_by[vname]), **st}
        if base_st:
            row["d_win"] = float(st.get("win") or 0) - float(base_st.get("win") or 0)
            row["d_add"] = float(st.get("add") or 0) - float(base_st.get("add") or 0)
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
    for name in score.sort_values("win", ascending=False)["variant"].head(6):
        if name in trade_dump:
            trade_dump[name].to_csv(out / f"trades_{name}.csv", index=False)

    keep = score[score.get("keep_edge") == True] if "keep_edge" in score.columns else score
    hit = keep[keep.get("hit_target_win") == True] if "hit_target_win" in keep.columns else keep.iloc[0:0]
    if len(hit):
        best = hit.sort_values(["win", "add"], ascending=[False, False]).iloc[0].to_dict()
        verdict = "WINRATE_LIFT"
    elif len(keep):
        best = keep.sort_values(["win", "add"], ascending=[False, False]).iloc[0].to_dict()
        verdict = (
            "WINRATE_PARTIAL"
            if float(best.get("d_win") or 0) > 0.02
            else "WINRATE_NO_LIFT"
        )
    else:
        best = score.sort_values("win", ascending=False).iloc[0].to_dict()
        verdict = "WINRATE_NO_LIFT"

    summary = {
        "baseline": base_st,
        "target_win": args.target_win,
        "verdict": verdict,
        "best": best,
        "note": "SMC/ICT-lite + VWAP on frozen stock_flow_opt base; Jul only; dual-window deferred.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\n=== verdict", verdict, flush=True)
    cols = [
        c
        for c in ["variant", "n", "win", "mean", "day_win", "add", "d_win", "d_add", "hit_target_win", "keep_edge"]
        if c in score.columns
    ]
    print(score.sort_values("win", ascending=False)[cols].to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)
    return 0 if verdict == "WINRATE_LIFT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
