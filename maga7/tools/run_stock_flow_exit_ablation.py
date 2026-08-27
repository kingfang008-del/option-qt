#!/usr/bin/env python3
"""Exit ablation for stock_flow_opt: easier TP / shorter time flatten.

Entry frozen (Jul pocket, rising):
  ret_60s<=-0.3%, dn_vol_share_120>=0.60  [+ optional vol_z>=1.5]
Exit grid only: tp × sl × max_hold_sec (time flatten = soft time stop).

Goal: lift per-trade win toward 55% while keeping mean>0, add>0, day_win>=0.55.

Example:
  PYTHONPATH=. python -m maga7.tools.run_stock_flow_exit_ablation \\
    --tag research_stock_flow_exit_ablation_jul10_23
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
# First ~1h after open (warm starts inside session at max(flow,disp)).
SESSIONS_OPEN1H = (("AM_0935_1030", "09:35", "10:30"),)

DISP = 0.003
FLOW = 120
SHARE = 0.60
DISP_SEC = 60


def _idx_at_or_before(ts_ns: np.ndarray, t_ns: int) -> int | None:
    i = int(np.searchsorted(ts_ns, t_ns, side="right") - 1)
    return i if i >= 0 else None


def _collect_arms(
    *,
    dates: list[str],
    symbols: list[str],
    stock_1s: Path,
    tick_root: Path,
    lock,
    otm: int,
    require_volz: float | None,
    stride_sec: int,
    rearm_gap_sec: int,
    sessions: tuple[tuple[str, str, str], ...] | None = None,
) -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []
    sess_list = sessions if sessions is not None else SESSIONS
    for date in dates:
        for sym in symbols:
            day = load_stock_1s_day(stock_1s, sym, date)
            arrays = prepare_smc_flow_day(day)
            if arrays is None:
                continue
            ts_ns, px = _stock_arrays(day)
            tday = load_option_tick_day(tick_root, sym, date)
            if tday is None or tday.empty:
                continue
            tpaths = _paths_by_ticker(tday)
            by_dte = lock.get((sym, date))
            if not by_dte or not tpaths:
                continue
            for sess_name, s0, s1 in sess_list:
                warm = max(FLOW, DISP_SEC, 120)
                t = pd.Timestamp(f"{date} {s0}:00", tz=NY) + pd.Timedelta(seconds=warm)
                t_end = pd.Timestamp(f"{date} {s1}:00", tz=NY)
                next_ok = t
                prev_on = False
                stride = pd.Timedelta(seconds=int(stride_sec))
                gap = pd.Timedelta(seconds=int(rearm_gap_sec))
                while t < t_end:
                    i = _idx_at_or_before(arrays["ts_ns"], int(to_ny(t).value))
                    on = False
                    if i is not None and i >= max(FLOW, DISP_SEC, 30):
                        share = dn_vol_share_at(arrays, i=i, window_sec=FLOW)
                        j0 = _idx_at_or_before(
                            arrays["ts_ns"],
                            int(arrays["ts_ns"][i]) - DISP_SEC * 1_000_000_000,
                        )
                        if share is not None and j0 is not None and j0 < i:
                            a = float(arrays["close"][j0])
                            b = float(arrays["close"][i])
                            if a > 0 and np.isfinite(a) and np.isfinite(b):
                                ret = b / a - 1.0
                                on = ret <= -DISP and share >= SHARE
                                if on and require_volz is not None:
                                    vz = (
                                        float(arrays["vol_z"][i])
                                        if "vol_z" in arrays
                                        and np.isfinite(arrays["vol_z"][i])
                                        else float("nan")
                                    )
                                    on = np.isfinite(vz) and vz >= float(require_volz)
                    fire = on and (t >= next_ok) and (not prev_on)
                    if fire:
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
                                arms.append(
                                    {
                                        "date": date,
                                        "symbol": sym,
                                        "session": sess_name,
                                        "arm_ts": to_ny(t),
                                        "ticker": ticker,
                                        "dte": dte,
                                        "pts": arr[0],
                                        "plast": arr[1],
                                    }
                                )
                                next_ok = t + gap
                    prev_on = on
                    t += stride
    return arms


def _score_arms(
    arms: list[dict[str, Any]],
    *,
    tp: float,
    sl: float,
    max_hold_sec: int,
    slip: float,
    position_frac: float,
    max_concurrent: int,
    cooldown_minutes: float,
) -> tuple[dict[str, Any], list[dict]]:
    raw: list[dict] = []
    for arm in arms:
        sim = simulate_trade_tpsl(
            arm["pts"],
            arm["plast"],
            arm["arm_ts"],
            tp=float(tp),
            sl=float(sl),
            max_hold_sec=int(max_hold_sec),
            slip=float(slip),
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
                "cell": f"tp{tp}_sl{sl}_h{max_hold_sec}",
                "event_source": "stock_flow_exit",
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
                position_frac=float(position_frac),
                max_concurrent=int(max_concurrent),
                cooldown_minutes=float(cooldown_minutes),
            )
        )
    return _stats(sized), sized


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_stock_flow_exit_ablation_jul10_23")
    ap.add_argument("--tick-root", default=str(DEFAULT_TICK_ROOT))
    ap.add_argument("--start-date", default="2026-07-10")
    ap.add_argument("--end-date", default="2026-07-23")
    ap.add_argument("--tp", default="0.08,0.10,0.12,0.15,0.20,0.25")
    ap.add_argument("--sl", default="0.12,0.15,0.20,0.25")
    ap.add_argument("--max-hold", default="120,180,300,450,600,900")
    ap.add_argument("--entries", default="baseline,volz15", help="baseline|volz15")
    ap.add_argument("--stride-sec", type=int, default=5)
    ap.add_argument("--rearm-gap-sec", type=int, default=60)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=4)
    ap.add_argument("--cooldown-minutes", type=float, default=1.0)
    ap.add_argument("--target-win", type=float, default=0.55)
    ap.add_argument("--min-n", type=int, default=40)
    args = ap.parse_args(argv)

    dates = [d for d in tick_dates(args.tick_root) if args.start_date <= d <= args.end_date]
    tps = [float(x) for x in args.tp.split(",") if x.strip()]
    sls = [float(x) for x in args.sl.split(",") if x.strip()]
    holds = [int(x) for x in args.max_hold.split(",") if x.strip()]
    entry_names = [x.strip() for x in args.entries.split(",") if x.strip()]

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    symbols = list(prof.get("symbols") or [])
    stock_1s = Path(paths.get("stock_1s_root") or "/mnt/s990/data/raw_1s/stocks").expanduser()
    tick_root = Path(args.tick_root)
    lock = load_multidte_lock_index(Path(paths["open_locked_map"]).expanduser())
    otm = resolve_otm_rungs(prof, default=3)
    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    print(
        f"exit ablation {dates[0]}..{dates[-1]} entries={entry_names} "
        f"grid tp={tps} sl={sls} hold={holds}",
        flush=True,
    )

    arms_cache: dict[str, list[dict]] = {}
    for en in entry_names:
        vz = 1.5 if en == "volz15" else None
        print(f"collecting arms entry={en}…", flush=True)
        arms_cache[en] = _collect_arms(
            dates=dates,
            symbols=symbols,
            stock_1s=stock_1s,
            tick_root=tick_root,
            lock=lock,
            otm=otm,
            require_volz=vz,
            stride_sec=int(args.stride_sec),
            rearm_gap_sec=int(args.rearm_gap_sec),
        )
        print(f"  arms={len(arms_cache[en])}", flush=True)

    rows: list[dict[str, Any]] = []
    trade_dump: dict[str, pd.DataFrame] = {}
    frozen_ref = None  # tp25/sl20/h900 baseline

    for en in entry_names:
        arms = arms_cache[en]
        for tp in tps:
            for sl in sls:
                for h in holds:
                    st, sized = _score_arms(
                        arms,
                        tp=tp,
                        sl=sl,
                        max_hold_sec=h,
                        slip=float(args.slip),
                        position_frac=float(args.position_frac),
                        max_concurrent=int(args.max_concurrent),
                        cooldown_minutes=float(args.cooldown_minutes),
                    )
                    name = f"{en}_tp{tp}_sl{sl}_h{h}"
                    row = {
                        "name": name,
                        "entry": en,
                        "tp": tp,
                        "sl": sl,
                        "max_hold_sec": h,
                        **st,
                    }
                    row["hit_target_win"] = bool(
                        int(st.get("n") or 0) >= int(args.min_n)
                        and float(st.get("win") or 0) >= float(args.target_win)
                    )
                    row["keep_edge"] = bool(
                        int(st.get("n") or 0) >= int(args.min_n)
                        and float(st.get("mean") or 0) > 0
                        and float(st.get("add") or 0) > 0
                        and float(st.get("day_win") or 0) >= 0.55
                    )
                    if en == "baseline" and tp == 0.25 and sl == 0.20 and h == 900:
                        frozen_ref = row
                    rows.append(row)
                    if sized and (
                        row["hit_target_win"]
                        or (tp == 0.25 and sl == 0.20 and h == 900)
                        or (row["keep_edge"] and float(st.get("win") or 0) >= 0.52)
                    ):
                        trade_dump[name] = pd.DataFrame(sized)

    score = pd.DataFrame(rows)
    score.to_csv(out / "scoreboard.csv", index=False)
    for i, name in enumerate(
        score.sort_values(["hit_target_win", "win", "add"], ascending=[False, False, False])[
            "name"
        ].head(12)
    ):
        if name in trade_dump:
            trade_dump[name].to_csv(out / f"trades_{i:02d}_{name}.csv", index=False)

    lift = score[score.hit_target_win & score.keep_edge].sort_values(
        ["win", "add"], ascending=[False, False]
    )
    partial = score[score.keep_edge].sort_values(["win", "add"], ascending=[False, False])
    if len(lift):
        best = lift.iloc[0].to_dict()
        verdict = "EXIT_WINRATE_LIFT"
    elif len(partial) and float(partial.iloc[0]["win"]) >= float(args.target_win) - 0.03:
        best = partial.iloc[0].to_dict()
        verdict = "EXIT_WINRATE_PARTIAL"
    elif len(partial):
        best = partial.iloc[0].to_dict()
        verdict = "EXIT_EDGE_OK_WIN_SHORT"
    else:
        best = score.sort_values("win", ascending=False).iloc[0].to_dict()
        verdict = "EXIT_NO_LIFT"

    # best by add among keep_edge for comparison
    best_add = (
        score[score.keep_edge].sort_values("add", ascending=False).iloc[0].to_dict()
        if score.keep_edge.any()
        else None
    )

    summary = {
        "frozen_ref_tp25_sl20_h900": frozen_ref,
        "target_win": args.target_win,
        "min_n": args.min_n,
        "verdict": verdict,
        "best_win_keep_edge": best,
        "best_add_keep_edge": best_add,
        "n_hit_target_and_edge": int(len(lift)),
        "note": "Entry frozen; exit grid only. Dual-window/quote deferred.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n=== verdict", verdict, "hit_n=", len(lift), flush=True)
    if frozen_ref:
        print(
            f"frozen ref: n={frozen_ref.get('n')} win={frozen_ref.get('win')} "
            f"add={frozen_ref.get('add')} day_win={frozen_ref.get('day_win')}",
            flush=True,
        )
    show = score.sort_values(
        ["hit_target_win", "win", "add"], ascending=[False, False, False]
    ).head(20)
    cols = [
        c
        for c in [
            "name",
            "n",
            "win",
            "mean",
            "day_win",
            "add",
            "frac_tp",
            "frac_sl",
            "frac_max_hold",
            "hit_target_win",
            "keep_edge",
        ]
        if c in show.columns
    ]
    print(show[cols].to_string(index=False), flush=True)
    if best:
        print(
            f"\nbest: {best.get('name')} win={best.get('win')} n={best.get('n')} "
            f"add={best.get('add')} day_win={best.get('day_win')}",
            flush=True,
        )
    print(f"wrote {out}", flush=True)
    return 0 if verdict == "EXIT_WINRATE_LIFT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
