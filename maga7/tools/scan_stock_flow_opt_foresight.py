#!/usr/bin/env python3
"""Stock order-flow proxy → buy ATM puts (foresight + causal validate).

Hypothesis: signal from *stock* 1s tape (ret + down-tick vol share / mf),
vehicle = ATM put priced on option tick prints.

Jul pocket (option tick days): foresight waves ask whether stock flow lifts
before dumps; causal TP/SL asks whether a simple stock-flow gate makes money.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_stock_flow_opt_foresight \\
    --tag research_stock_flow_opt_jul10_23
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
from maga7.tools.scan_am_certainty_morph_tpsl import _ok, _stats
from maga7.tools.scan_session_horizon_foresight import (
    _fwd_trade_rets_arr,
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


def _idx_at_or_before(ts_ns: np.ndarray, t_ns: int) -> int | None:
    i = int(np.searchsorted(ts_ns, t_ns, side="right") - 1)
    return i if i >= 0 else None


def _stock_feat_at(
    arrays: dict[str, Any],
    *,
    t: pd.Timestamp,
    flow_sec: int,
    disp_sec: int,
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
    return {
        "dn_vol_share": float(share),
        "stock_ret_disp": float(b / a - 1.0),
        "mf": mf,
        "streak_dn": float(sd),
    }


def _find_dump_waves(
    ts_ns: np.ndarray,
    px: np.ndarray,
    *,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
    horizon_sec: int,
    wave_thr: float,
    stride_sec: int,
    min_gap_sec: int,
) -> list[dict[str, Any]]:
    t0 = to_ny(t_start)
    t1 = to_ny(t_end)
    stride = pd.Timedelta(seconds=int(stride_sec))
    gap = pd.Timedelta(seconds=int(min_gap_sec))
    h_ns = int(horizon_sec) * 1_000_000_000
    out: list[dict[str, Any]] = []
    t = t0
    next_ok = t0
    while t < t1:
        if t >= next_ok:
            i0 = _idx_at_or_before(ts_ns, int(t.value))
            if i0 is not None:
                end_ns = int(ts_ns[i0]) + h_ns
                i1 = int(np.searchsorted(ts_ns, end_ns, side="right") - 1)
                if i1 > i0:
                    a = float(px[i0])
                    win = px[i0 : i1 + 1]
                    if a > 0 and np.isfinite(a):
                        depth = float(np.nanmin(win) / a - 1.0)
                        if depth <= -float(wave_thr):
                            out.append({"t0": t, "stock_depth": depth})
                            next_ok = t + gap
        t += stride
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_stock_flow_opt_jul10_23")
    ap.add_argument("--tick-root", default=str(DEFAULT_TICK_ROOT))
    ap.add_argument("--wave-thr", default="0.005,0.008,0.012")
    ap.add_argument("--horizons", default="300,600,900")
    ap.add_argument("--flow-sec", default="60,120")
    ap.add_argument("--disp-sec", type=int, default=60)
    ap.add_argument("--disp-thr", default="0.003,0.005,0.008")
    ap.add_argument("--min-dn-share", default="0.55,0.60,0.65")
    ap.add_argument("--tp", default="0.15,0.20,0.25")
    ap.add_argument("--sl", default="0.15,0.20")
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--stride-sec", type=int, default=5)
    ap.add_argument("--rearm-gap-sec", type=int, default=60)
    ap.add_argument("--fire-mode", default="rising", choices=("rising", "first", "hold"))
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=4)
    ap.add_argument("--cooldown-minutes", type=float, default=1.0)
    ap.add_argument("--min-n", type=int, default=15)
    ap.add_argument("--min-day-win", type=float, default=0.55)
    ap.add_argument(
        "--sessions",
        default="AM_0935_1030,CORE_1030_1200,MID_1200_1400,PM_1400_1530",
    )
    args = ap.parse_args(argv)

    tick_root = Path(args.tick_root)
    dates = tick_dates(tick_root)
    if not dates:
        print("no tick dates", flush=True)
        return 2

    wave_thrs = [float(x) for x in args.wave_thr.split(",") if x.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    flow_secs = [int(x) for x in args.flow_sec.split(",") if x.strip()]
    disp_thrs = [float(x) for x in args.disp_thr.split(",") if x.strip()]
    shares = [float(x) for x in args.min_dn_share.split(",") if x.strip()]
    tps = [float(x) for x in args.tp.split(",") if x.strip()]
    sls = [float(x) for x in args.sl.split(",") if x.strip()]
    want_sess = {x.strip() for x in args.sessions.split(",") if x.strip()}
    sessions = tuple(s for s in SESSIONS if s[0] in want_sess)

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    symbols = list(prof.get("symbols") or [])
    stock_1s = Path(paths.get("stock_1s_root") or "/mnt/s990/data/raw_1s/stocks").expanduser()
    lock = load_multidte_lock_index(Path(paths["open_locked_map"]).expanduser())
    otm = resolve_otm_rungs(prof, default=3)
    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    print(
        f"stock_flow→opt {dates[0]}..{dates[-1]} days={len(dates)} "
        f"fire={args.fire_mode}",
        flush=True,
    )

    # ---------- foresight wave features ----------
    wave_rows: list[dict[str, Any]] = []
    ctrl_rows: list[dict[str, Any]] = []
    # ---------- causal arms ----------
    arms: list[dict[str, Any]] = []

    for di, date in enumerate(dates):
        if di % 2 == 0:
            print(
                f"[day] {date} ({di+1}/{len(dates)}) waves={len(wave_rows)} arms={len(arms)}",
                flush=True,
            )
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

            for sess_name, s0, s1 in sessions:
                t_start = pd.Timestamp(f"{date} {s0}:00", tz=NY)
                t_end = pd.Timestamp(f"{date} {s1}:00", tz=NY)
                warm = max(max(flow_secs), int(args.disp_sec), 120)

                # foresight waves
                for H in horizons:
                    for wthr in wave_thrs:
                        waves = _find_dump_waves(
                            ts_ns,
                            px,
                            t_start=t_start + pd.Timedelta(seconds=warm),
                            t_end=t_end - pd.Timedelta(seconds=H),
                            horizon_sec=H,
                            wave_thr=wthr,
                            stride_sec=30,
                            min_gap_sec=300,
                        )
                        for w in waves:
                            t0 = w["t0"]
                            spot = _spot_at_arr(ts_ns, px, t0)
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
                            if not ticker:
                                continue
                            arr = tpaths.get(str(ticker).replace("O:", ""))
                            if arr is None:
                                continue
                            fwd = _fwd_trade_rets_arr(
                                arr[0], arr[1], t0, [H], slip=float(args.slip)
                            )
                            if not fwd:
                                continue
                            for fw in flow_secs:
                                feat = _stock_feat_at(
                                    arrays, t=t0, flow_sec=fw, disp_sec=int(args.disp_sec)
                                )
                                if feat is None:
                                    continue
                                wave_rows.append(
                                    {
                                        "date": date,
                                        "symbol": sym,
                                        "session": sess_name,
                                        "kind": "wave",
                                        "horizon_sec": H,
                                        "wave_thr": wthr,
                                        "flow_sec": fw,
                                        "stock_depth": w["stock_depth"],
                                        "oracle_ret": fwd[0]["oracle_ret"],
                                        "clock_ret": fwd[0]["clock_ret"],
                                        **feat,
                                    }
                                )

                # controls
                t = t_start + pd.Timedelta(seconds=warm)
                t_lim = t_end - pd.Timedelta(seconds=max(horizons))
                while t < t_lim:
                    for fw in flow_secs:
                        feat = _stock_feat_at(
                            arrays, t=t, flow_sec=fw, disp_sec=int(args.disp_sec)
                        )
                        if feat is None:
                            continue
                        ctrl_rows.append(
                            {
                                "date": date,
                                "symbol": sym,
                                "session": sess_name,
                                "kind": "control",
                                "flow_sec": fw,
                                **feat,
                            }
                        )
                    t += pd.Timedelta(seconds=300)

                # causal arms: stock dump + dn_vol_share → ATM put
                for dthr in disp_thrs:
                    for fw in flow_secs:
                        for sh in shares:
                            sk = f"stk_d{dthr}_f{fw}_sh{sh}"
                            t = t_start + pd.Timedelta(seconds=warm)
                            next_ok = t
                            prev_on = False
                            stride = pd.Timedelta(seconds=int(args.stride_sec))
                            gap = pd.Timedelta(seconds=int(args.rearm_gap_sec))
                            while t < t_end:
                                feat = _stock_feat_at(
                                    arrays,
                                    t=t,
                                    flow_sec=fw,
                                    disp_sec=int(args.disp_sec),
                                )
                                on = bool(
                                    feat is not None
                                    and feat["stock_ret_disp"] <= -float(dthr)
                                    and feat["dn_vol_share"] >= float(sh)
                                )
                                fire = False
                                if on and t >= next_ok:
                                    if args.fire_mode == "hold":
                                        fire = True
                                    elif args.fire_mode == "first":
                                        fire = not prev_on
                                    else:
                                        fire = not prev_on  # rising
                                if fire and feat is not None:
                                    if args.fire_mode == "first" and any(
                                        a["date"] == date
                                        and a["symbol"] == sym
                                        and a["session"] == sess_name
                                        and a["spec_key"] == sk
                                        for a in arms
                                    ):
                                        fire = False
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
                                            arr = tpaths.get(
                                                str(ticker).replace("O:", "")
                                            )
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
                                                        "disp_thr": float(dthr),
                                                        "flow_sec": int(fw),
                                                        "min_dn_share": float(sh),
                                                        "spec_key": sk,
                                                        **feat,
                                                    }
                                                )
                                                next_ok = t + gap
                                                if args.fire_mode == "first":
                                                    break
                                prev_on = on
                                t += stride
                            if args.fire_mode == "first":
                                # continue outer specs
                                pass

    waves_df = pd.DataFrame(wave_rows)
    ctrl_df = pd.DataFrame(ctrl_rows)
    waves_df.to_csv(out / "waves.csv", index=False)
    ctrl_df.to_csv(out / "controls.csv", index=False)

    # foresight scoreboard
    f_rows: list[dict[str, Any]] = []
    for H in horizons:
        for wthr in wave_thrs:
            for fw in flow_secs:
                w = waves_df[
                    (waves_df.horizon_sec == H)
                    & (waves_df.wave_thr == wthr)
                    & (waves_df.flow_sec == fw)
                ]
                c = ctrl_df[ctrl_df.flow_sec == fw]
                if w.empty:
                    continue
                row: dict[str, Any] = {
                    "horizon_sec": H,
                    "wave_thr": wthr,
                    "flow_sec": fw,
                    "wave_n": int(len(w)),
                    "wave_dn_share_mean": float(w["dn_vol_share"].mean()),
                    "wave_ret_mean": float(w["stock_ret_disp"].mean()),
                    "wave_frac_share_ge_0.55": float((w["dn_vol_share"] >= 0.55).mean()),
                    "wave_frac_share_ge_0.60": float((w["dn_vol_share"] >= 0.60).mean()),
                    "wave_oracle_mean": float(w["oracle_ret"].mean()),
                    "ctrl_n": int(len(c)),
                    "ctrl_frac_share_ge_0.55": (
                        float((c["dn_vol_share"] >= 0.55).mean()) if len(c) else None
                    ),
                }
                if row["ctrl_frac_share_ge_0.55"]:
                    row["lift_share_0.55"] = row["wave_frac_share_ge_0.55"] / row[
                        "ctrl_frac_share_ge_0.55"
                    ]
                else:
                    row["lift_share_0.55"] = None
                lift = row["lift_share_0.55"]
                row["distill_ok"] = bool(
                    row["wave_n"] >= 15
                    and lift is not None
                    and float(lift) >= 1.2
                    and row["wave_oracle_mean"] > 0
                    and row["wave_frac_share_ge_0.55"] >= 0.45
                )
                f_rows.append(row)
    fscore = pd.DataFrame(f_rows)
    fscore.to_csv(out / "foresight_scoreboard.csv", index=False)
    distill_n = int(fscore["distill_ok"].sum()) if len(fscore) else 0
    foresight_verdict = "FORESIGHT_DISTILL" if distill_n else "FORESIGHT_NO_DISTILL"

    # causal TP/SL
    print(f"causal arms={len(arms)}; scoring…", flush=True)
    cells: list[dict[str, Any]] = []
    specs = sorted({a["spec_key"] for a in arms})
    for sk in specs:
        sample = next(a for a in arms if a["spec_key"] == sk)
        for tp in tps:
            for sl in sls:
                cells.append(
                    {
                        "name": f"{sk}_tp{tp}_sl{sl}",
                        "spec_key": sk,
                        "disp_thr": sample["disp_thr"],
                        "flow_sec": sample["flow_sec"],
                        "min_dn_share": sample["min_dn_share"],
                        "tp": float(tp),
                        "sl": float(sl),
                    }
                )

    score_rows: list[dict[str, Any]] = []
    win_pass: list[dict[str, Any]] = []
    trade_dump: dict[str, pd.DataFrame] = {}
    for ci, cell in enumerate(cells):
        if ci % 30 == 0:
            print(f"[cell] {ci+1}/{len(cells)} pass={len(win_pass)}", flush=True)
        raw: list[dict] = []
        for arm in arms:
            if arm["spec_key"] != cell["spec_key"]:
                continue
            sim = simulate_trade_tpsl(
                arm["pts"],
                arm["plast"],
                arm["arm_ts"],
                tp=float(cell["tp"]),
                sl=float(cell["sl"]),
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
                    "cell": cell["name"],
                    "event_source": "stock_flow_opt",
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
        ok = _ok(st, min_n=int(args.min_n), min_day_win=float(args.min_day_win))
        row = {**cell, "window_pass": ok, **st}
        score_rows.append(row)
        if ok:
            win_pass.append(row)
            trade_dump[cell["name"]] = pd.DataFrame(sized)
            print(
                f"  *** PASS {cell['name']} n={st.get('n')} mean={st.get('mean'):+.3f} "
                f"day_win={st.get('day_win')} add={st.get('add'):+.3f}",
                flush=True,
            )

    score = pd.DataFrame(score_rows)
    score.to_csv(out / "causal_scoreboard.csv", index=False)
    win_pass = sorted(win_pass, key=lambda r: float(r.get("add") or 0), reverse=True)
    for i, p in enumerate(win_pass[:10]):
        name = p["name"]
        if name in trade_dump and len(trade_dump[name]):
            trade_dump[name].to_csv(out / f"trades_pass{i:02d}_{name}.csv", index=False)

    causal_verdict = "VALIDATE_PASS" if win_pass else "VALIDATE_REJECT"
    summary = {
        "expert_kind": "stock_flow_opt",
        "hypothesis": "stock 1s OF proxy (ret+dn_vol_share) → ATM put on option tick",
        "dates": dates,
        "fire_mode": args.fire_mode,
        "foresight_verdict": foresight_verdict,
        "foresight_distill_n": distill_n,
        "causal_verdict": causal_verdict,
        "causal_pass_n": len(win_pass),
        "n_arms": len(arms),
        "champion": win_pass[0] if win_pass else None,
        "best_foresight": (
            fscore.sort_values(
                ["distill_ok", "lift_share_0.55", "wave_oracle_mean"],
                ascending=[False, False, False],
            )
            .head(1)
            .to_dict("records")[0]
            if len(fscore)
            else None
        ),
        "note": (
            "Stock tape still has no aggressor; dn_vol_share is Δclose×volume proxy. "
            "Single Jul window only."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n=== foresight", foresight_verdict, "distill_n=", distill_n, flush=True)
    if len(fscore):
        cols = [
            c
            for c in [
                "horizon_sec",
                "wave_thr",
                "flow_sec",
                "wave_n",
                "wave_dn_share_mean",
                "wave_frac_share_ge_0.55",
                "ctrl_frac_share_ge_0.55",
                "lift_share_0.55",
                "wave_oracle_mean",
                "distill_ok",
            ]
            if c in fscore.columns
        ]
        print(
            fscore.sort_values(
                ["distill_ok", "lift_share_0.55"], ascending=[False, False]
            )
            .head(12)[cols]
            .to_string(index=False),
            flush=True,
        )
    print("\n=== causal", causal_verdict, "pass_n=", len(win_pass), flush=True)
    if win_pass:
        c = win_pass[0]
        print(
            f"champion {c['name']}: n={c.get('n')} mean={c.get('mean')} "
            f"day_win={c.get('day_win')} add={c.get('add')}",
            flush=True,
        )
    elif len(score):
        near = score.sort_values("add", ascending=False).head(10)
        print(
            near[["name", "n", "mean", "day_win", "add"]].to_string(index=False),
            flush=True,
        )
    print(f"wrote {out}", flush=True)
    return 0 if (distill_n or win_pass) else 1


if __name__ == "__main__":
    raise SystemExit(main())
