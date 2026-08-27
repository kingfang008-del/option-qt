#!/usr/bin/env python3
"""UP / continuation foresight for quiet-bull windows (stock flow → ATM call).

Protocol (no Jul in selection):
  discover  Feb1–Apr30  → foresight distill only here
  holdout   Jun1–Jun30  → report only
  blind     Jul10–Jul23 → report only

Hypothesis: before stock *rally* waves, up-tick vol share / positive short ret
lifts vs control; if distill holds on discover, causal UP gate may fit 2–6.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_stock_flow_up_foresight \\
    --tag research_stock_flow_up_foresight_feb_jun_jul
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

WINDOWS = (
    ("discover", "2026-02-01", "2026-04-30"),
    ("holdout", "2026-06-01", "2026-06-30"),
    ("blind", "2026-07-10", "2026-07-23"),
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
    dn_share = dn_vol_share_at(arrays, i=i, window_sec=int(flow_sec))
    if dn_share is None:
        return None
    j0 = _idx_at_or_before(ts_ns, int(ts_ns[i]) - int(disp_sec) * 1_000_000_000)
    if j0 is None or j0 >= i:
        return None
    a, b = float(c[j0]), float(c[i])
    if a <= 0 or not np.isfinite(a) or not np.isfinite(b):
        return None
    mf = float(arrays["mf"][i]) if np.isfinite(arrays["mf"][i]) else float("nan")
    return {
        "dn_vol_share": float(dn_share),
        "up_vol_share": float(1.0 - dn_share),
        "stock_ret_disp": float(b / a - 1.0),
        "mf": mf,
    }


def _find_rally_waves(
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
                        height = float(np.nanmax(win) / a - 1.0)
                        if height >= float(wave_thr):
                            out.append({"t0": t, "stock_height": height})
                            next_ok = t + gap
        t += stride
    return out


def _foresight_board(waves: pd.DataFrame, ctrl: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if waves.empty:
        return pd.DataFrame(rows)
    for (H, wthr, fw), w in waves.groupby(["horizon_sec", "wave_thr", "flow_sec"]):
        c = ctrl[ctrl.flow_sec == fw] if len(ctrl) else ctrl
        row: dict[str, Any] = {
            "horizon_sec": int(H),
            "wave_thr": float(wthr),
            "flow_sec": int(fw),
            "wave_n": int(len(w)),
            "wave_up_share_mean": float(w["up_vol_share"].mean()),
            "wave_ret_mean": float(w["stock_ret_disp"].mean()),
            "wave_frac_up_ge_0.55": float((w["up_vol_share"] >= 0.55).mean()),
            "wave_frac_up_ge_0.60": float((w["up_vol_share"] >= 0.60).mean()),
            "wave_oracle_mean": float(w["oracle_ret"].mean()),
            "ctrl_n": int(len(c)),
            "ctrl_frac_up_ge_0.55": (
                float((c["up_vol_share"] >= 0.55).mean()) if len(c) else None
            ),
        }
        if row["ctrl_frac_up_ge_0.55"]:
            row["lift_up_0.55"] = row["wave_frac_up_ge_0.55"] / row["ctrl_frac_up_ge_0.55"]
        else:
            row["lift_up_0.55"] = None
        lift = row["lift_up_0.55"]
        row["distill_ok"] = bool(
            row["wave_n"] >= 15
            and lift is not None
            and float(lift) >= 1.2
            and row["wave_oracle_mean"] > 0
            and row["wave_frac_up_ge_0.55"] >= 0.45
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _collect_window(
    *,
    dates: list[str],
    symbols: list[str],
    stock_1s: Path,
    tick_root: Path,
    lock,
    otm: int,
    sessions: tuple,
    horizons: list[int],
    wave_thrs: list[float],
    flow_secs: list[int],
    disp_sec: int,
    disp_thrs: list[float],
    shares: list[float],
    slip: float,
    stride_sec: int,
    rearm_gap_sec: int,
    fire_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    wave_rows: list[dict[str, Any]] = []
    ctrl_rows: list[dict[str, Any]] = []
    arms: list[dict[str, Any]] = []

    for di, date in enumerate(dates):
        if di % 5 == 0:
            print(
                f"  [day] {date} ({di+1}/{len(dates)}) waves={len(wave_rows)} arms={len(arms)}",
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
                warm = max(max(flow_secs), int(disp_sec), 120)

                for H in horizons:
                    for wthr in wave_thrs:
                        waves = _find_rally_waves(
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
                                direction="UP",
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
                                arr[0], arr[1], t0, [H], slip=float(slip)
                            )
                            if not fwd:
                                continue
                            for fw in flow_secs:
                                feat = _stock_feat_at(
                                    arrays, t=t0, flow_sec=fw, disp_sec=int(disp_sec)
                                )
                                if feat is None:
                                    continue
                                wave_rows.append(
                                    {
                                        "date": date,
                                        "symbol": sym,
                                        "session": sess_name,
                                        "horizon_sec": H,
                                        "wave_thr": wthr,
                                        "flow_sec": fw,
                                        "stock_height": w["stock_height"],
                                        "oracle_ret": fwd[0]["oracle_ret"],
                                        "clock_ret": fwd[0]["clock_ret"],
                                        **feat,
                                    }
                                )

                # controls (stride 60s to keep size down)
                t = t_start + pd.Timedelta(seconds=warm)
                t_lim = t_end - pd.Timedelta(seconds=max(horizons))
                while t < t_lim:
                    for fw in flow_secs:
                        feat = _stock_feat_at(
                            arrays, t=t, flow_sec=fw, disp_sec=int(disp_sec)
                        )
                        if feat is not None:
                            ctrl_rows.append(
                                {"date": date, "symbol": sym, "flow_sec": fw, **feat}
                            )
                    t += pd.Timedelta(seconds=60)

                # causal rising UP gate — independent state per spec
                stride = pd.Timedelta(seconds=int(stride_sec))
                gap = pd.Timedelta(seconds=int(rearm_gap_sec))
                for dthr in disp_thrs:
                    for fw in flow_secs:
                        for sh in shares:
                            t = t_start + pd.Timedelta(seconds=warm)
                            next_ok = t
                            prev_on = False
                            sk = f"up_d{dthr}_f{fw}_sh{sh}"
                            while t < t_end:
                                feat = _stock_feat_at(
                                    arrays, t=t, flow_sec=fw, disp_sec=int(disp_sec)
                                )
                                on = bool(
                                    feat is not None
                                    and feat["stock_ret_disp"] >= float(dthr)
                                    and feat["up_vol_share"] >= float(sh)
                                )
                                if fire_mode == "rising":
                                    fire = on and (not prev_on) and (t >= next_ok)
                                else:
                                    fire = on and (t >= next_ok)
                                if fire:
                                    spot = _spot_at_arr(ts_ns, px, t)
                                    ticker, dte, _ = resolve_open_lock_contract(
                                        by_dte,
                                        direction="UP",
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
                                                    "spec_key": sk,
                                                    "disp_thr": dthr,
                                                    "flow_sec": fw,
                                                    "min_up_share": sh,
                                                }
                                            )
                                            next_ok = t + gap
                                prev_on = on
                                t += stride

    return pd.DataFrame(wave_rows), pd.DataFrame(ctrl_rows), arms


def _score_causal(
    arms: list[dict[str, Any]],
    *,
    tps: list[float],
    sls: list[float],
    max_hold_sec: int,
    slip: float,
    position_frac: float,
    max_concurrent: int,
    cooldown_minutes: float,
    min_n: int,
    min_day_win: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    specs = sorted({a["spec_key"] for a in arms})
    score_rows: list[dict[str, Any]] = []
    win_pass: list[dict[str, Any]] = []
    for sk in specs:
        sample = next(a for a in arms if a["spec_key"] == sk)
        for tp in tps:
            for sl in sls:
                raw: list[dict] = []
                for arm in arms:
                    if arm["spec_key"] != sk:
                        continue
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
                            "dir": "UP",
                            "session": arm["session"],
                            "entry_ts": str(et),
                            "exit_ts": str(et + pd.Timedelta(seconds=sim["hold_sec"])),
                            "ticker": arm["ticker"],
                            "ret": sim["ret"],
                            "exit_reason": sim["reason"],
                            "hold_sec": sim["hold_sec"],
                            "cell": f"{sk}_tp{tp}_sl{sl}",
                            "event_source": "stock_flow_up",
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
                st = _stats(sized)
                ok = _ok(st, min_n=int(min_n), min_day_win=float(min_day_win))
                row = {
                    "name": f"{sk}_tp{tp}_sl{sl}",
                    "spec_key": sk,
                    "disp_thr": sample["disp_thr"],
                    "flow_sec": sample["flow_sec"],
                    "min_up_share": sample["min_up_share"],
                    "tp": tp,
                    "sl": sl,
                    "window_pass": ok,
                    **st,
                }
                score_rows.append(row)
                if ok:
                    win_pass.append(row)
    win_pass = sorted(win_pass, key=lambda r: float(r.get("add") or 0), reverse=True)
    return pd.DataFrame(score_rows), win_pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_stock_flow_up_foresight_feb_jun_jul")
    ap.add_argument("--tick-root", default=str(DEFAULT_TICK_ROOT))
    ap.add_argument("--wave-thr", default="0.005,0.008,0.012")
    ap.add_argument("--horizons", default="300,600,900")
    ap.add_argument("--flow-sec", default="60,120")
    ap.add_argument("--disp-sec", type=int, default=60)
    ap.add_argument("--disp-thr", default="0.003,0.005")
    ap.add_argument("--min-up-share", default="0.55,0.60")
    ap.add_argument("--tp", default="0.10,0.15,0.20")
    ap.add_argument("--sl", default="0.15,0.20,0.25")
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--stride-sec", type=int, default=5)
    ap.add_argument("--rearm-gap-sec", type=int, default=60)
    ap.add_argument("--fire-mode", default="rising")
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=4)
    ap.add_argument("--cooldown-minutes", type=float, default=1.0)
    ap.add_argument("--min-n", type=int, default=25)
    ap.add_argument("--min-day-win", type=float, default=0.55)
    args = ap.parse_args(argv)

    wave_thrs = [float(x) for x in args.wave_thr.split(",") if x.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    flow_secs = [int(x) for x in args.flow_sec.split(",") if x.strip()]
    disp_thrs = [float(x) for x in args.disp_thr.split(",") if x.strip()]
    shares = [float(x) for x in args.min_up_share.split(",") if x.strip()]
    tps = [float(x) for x in args.tp.split(",") if x.strip()]
    sls = [float(x) for x in args.sl.split(",") if x.strip()]

    all_tick = tick_dates(args.tick_root)
    prof = load_profile(args.profile)
    paths = prof["_paths"]
    symbols = list(prof.get("symbols") or [])
    stock_1s = Path(paths.get("stock_1s_root") or "/mnt/s990/data/raw_1s/stocks").expanduser()
    tick_root = Path(args.tick_root)
    lock = load_multidte_lock_index(Path(paths["open_locked_map"]).expanduser())
    otm = resolve_otm_rungs(prof, default=3)
    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    by_win: dict[str, Any] = {}
    for wname, a, b in WINDOWS:
        dates = [d for d in all_tick if a <= d <= b]
        print(f"\n=== window {wname} {a}..{b} n_dates={len(dates)} ===", flush=True)
        if not dates:
            by_win[wname] = {"error": "empty"}
            continue
        waves, ctrl, arms = _collect_window(
            dates=dates,
            symbols=symbols,
            stock_1s=stock_1s,
            tick_root=tick_root,
            lock=lock,
            otm=otm,
            sessions=SESSIONS,
            horizons=horizons,
            wave_thrs=wave_thrs,
            flow_secs=flow_secs,
            disp_sec=int(args.disp_sec),
            disp_thrs=disp_thrs,
            shares=shares,
            slip=float(args.slip),
            stride_sec=int(args.stride_sec),
            rearm_gap_sec=int(args.rearm_gap_sec),
            fire_mode=str(args.fire_mode),
        )
        waves.to_csv(out / f"waves_{wname}.csv", index=False)
        ctrl.to_csv(out / f"controls_{wname}.csv", index=False)
        fscore = _foresight_board(waves, ctrl)
        fscore.to_csv(out / f"foresight_scoreboard_{wname}.csv", index=False)
        distill_n = int(fscore["distill_ok"].sum()) if len(fscore) else 0
        best_f = (
            fscore.sort_values(
                ["distill_ok", "lift_up_0.55", "wave_oracle_mean"],
                ascending=[False, False, False],
            )
            .head(1)
            .to_dict("records")[0]
            if len(fscore)
            else None
        )
        print(
            f"  foresight distill_n={distill_n} waves={len(waves)} arms={len(arms)}",
            flush=True,
        )
        if best_f:
            print(
                f"  best_f: H={best_f.get('horizon_sec')} thr={best_f.get('wave_thr')} "
                f"lift={best_f.get('lift_up_0.55')} oracle={best_f.get('wave_oracle_mean')} "
                f"ok={best_f.get('distill_ok')}",
                flush=True,
            )

        cscore, cpass = _score_causal(
            arms,
            tps=tps,
            sls=sls,
            max_hold_sec=int(args.max_hold_sec),
            slip=float(args.slip),
            position_frac=float(args.position_frac),
            max_concurrent=int(args.max_concurrent),
            cooldown_minutes=float(args.cooldown_minutes),
            min_n=int(args.min_n),
            min_day_win=float(args.min_day_win),
        )
        cscore.to_csv(out / f"causal_scoreboard_{wname}.csv", index=False)
        print(
            f"  causal pass_n={len(cpass)} "
            + (
                f"champ={cpass[0]['name']} add={cpass[0].get('add')} win={cpass[0].get('win')}"
                if cpass
                else (
                    f"best_add={cscore.sort_values('add', ascending=False).iloc[0].to_dict() if len(cscore) else None}"
                )
            ),
            flush=True,
        )
        by_win[wname] = {
            "n_dates": len(dates),
            "dates": dates,
            "n_waves": int(len(waves)),
            "n_arms": int(len(arms)),
            "foresight_verdict": (
                "FORESIGHT_DISTILL" if distill_n else "FORESIGHT_NO_DISTILL"
            ),
            "foresight_distill_n": distill_n,
            "best_foresight": best_f,
            "causal_verdict": "VALIDATE_PASS" if cpass else "VALIDATE_REJECT",
            "causal_pass_n": len(cpass),
            "champion": cpass[0] if cpass else None,
            "best_causal_add": (
                cscore.sort_values("add", ascending=False).head(1).to_dict("records")[0]
                if len(cscore)
                else None
            ),
        }

    disc = by_win.get("discover", {})
    hold = by_win.get("holdout", {})
    blind = by_win.get("blind", {})
    if disc.get("foresight_verdict") == "FORESIGHT_DISTILL":
        # transfer: same distill cells reported on hold/blind (already per-window)
        if hold.get("foresight_verdict") == "FORESIGHT_DISTILL":
            overall = "DISCOVER_HOLDOUT_DISTILL"
        else:
            overall = "DISCOVER_ONLY_DISTILL"
        if disc.get("causal_verdict") == "VALIDATE_PASS" and hold.get(
            "causal_verdict"
        ) == "VALIDATE_PASS":
            overall = "OOS_CAUSAL_PASS_" + overall
    else:
        overall = "NO_DISTILL_ON_DISCOVER"

    summary = {
        "expert_kind": "stock_flow_up",
        "hypothesis": "stock 1s up_vol_share + positive ret → ATM call (quiet-bull)",
        "protocol": {
            "discover": "2026-02-01..2026-04-30 (selection)",
            "holdout": "2026-06-01..2026-06-30 (report)",
            "blind": "2026-07-10..2026-07-23 (blind)",
        },
        "overall_verdict": overall,
        "windows": by_win,
        "note": (
            "Selection uses discover foresight only. May omitted as bridge. "
            "Jul not used for distill/cell pick."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n=== OVERALL", overall, "===", flush=True)
    for wname in ("discover", "holdout", "blind"):
        w = by_win.get(wname, {})
        print(
            f"{wname}: foresight={w.get('foresight_verdict')} distill_n={w.get('foresight_distill_n')} "
            f"causal={w.get('causal_verdict')} pass_n={w.get('causal_pass_n')}",
            flush=True,
        )
    print(f"wrote {out}", flush=True)
    return 0 if str(overall).startswith("DISCOVER") or "OOS_CAUSAL" in str(overall) else 1


if __name__ == "__main__":
    raise SystemExit(main())
