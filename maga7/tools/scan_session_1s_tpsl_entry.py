#!/usr/bin/env python3
"""Session entry on causal 1s features; exit via option-path TP/SL (not clock H).

Primary exit = first passage of +tp / −sl on trade-last MTM (after slip).
``max_hold_sec`` is safety-only (flatten), not the research thesis.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_session_1s_tpsl_entry \\
    --start-date 2026-05-01 --end-date 2026-07-22 \\
    --tag research_session_1s_tpsl_may_jul
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
from maga7.common.option_trade_tpsl import simulate_trade_tpsl
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import to_ny
from maga7.common.session_1s_features import features_at, prepare_day_arrays
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_session_1s_feature_entry import RULES
from maga7.tools.scan_session_horizon_foresight import SESSIONS, _bdates, _paths_by_ticker

NY = "America/New_York"
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")
DEFAULT_STOCK_1S = Path("/mnt/s990/data/raw_1s/stocks")
FREEZE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)


def _port(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "mean": None, "win": None, "add": 0.0, "day_win": None, "red_days": 0}
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(str(r["date"]), []).append(r)
    sized: list[dict] = []
    for d in sorted(by):
        sized.extend(
            _portfolio_day(by[d], position_frac=0.10, max_concurrent=2, cooldown_minutes=2.0)
        )
    if not sized:
        return {"n": 0, "mean": None, "win": None, "add": 0.0, "day_win": None, "red_days": 0}
    t = pd.DataFrame(sized)
    t["pnl_frac"] = t["ret"].astype(float) * t["size"].astype(float)
    day = t.groupby("date")["pnl_frac"].sum()
    reasons = pd.Series([r.get("exit_reason") for r in sized])
    return {
        "n": int(len(t)),
        "mean": float(t["ret"].mean()),
        "win": float((t["ret"] > 0).mean()),
        "add": float(t["pnl_frac"].sum()),
        "day_win": float((day > 0).mean()),
        "red_days": int((day < 0).sum()),
        "tpd": float(len(t) / max(t["date"].nunique(), 1)),
        "worst_day": float(day.min()),
        "frac_tp": float((reasons == "tp").mean()) if len(reasons) else None,
        "frac_sl": float((reasons == "sl").mean()) if len(reasons) else None,
        "frac_max_hold": float((reasons == "max_hold").mean()) if len(reasons) else None,
        "hold_p50": float(pd.Series([r.get("hold_sec") for r in sized]).median()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=FREEZE)
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--tag", default="research_session_1s_tpsl")
    ap.add_argument("--stock-1s-root", default=str(DEFAULT_STOCK_1S))
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--stride-sec", type=int, default=60)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--tps", default="0.05,0.10,0.15,0.20")
    ap.add_argument("--sls", default="0.05,0.08,0.10,0.15")
    ap.add_argument("--sessions", default="AM_0930_1000,MID_1230_1330")
    ap.add_argument(
        "--rules",
        default="MOM60,MOM60_MF,MOM60_VOLR15,MOM60_VOLZ15,MOM60_VWAP,FADE_VWAP35",
    )
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    symbols = list(prof.get("symbols") or [])
    tps = [float(x) for x in args.tps.split(",") if x.strip()]
    sls = [float(x) for x in args.sls.split(",") if x.strip()]
    want_sess = {x.strip() for x in args.sessions.split(",") if x.strip()}
    active = tuple(s for s in SESSIONS if s[0] in want_sess)
    want_rules = [x.strip() for x in args.rules.split(",") if x.strip() in RULES]
    stock_1s = Path(args.stock_1s_root)
    trades_root = Path(args.trades_root)
    dates = _bdates(args.start_date, args.end_date)

    lock_path = Path(paths.get("open_locked_map") or paths.get("locked_map")).expanduser()
    multi_idx = load_multidte_lock_index(lock_path) if lock_path.is_file() else {}
    otm_rungs = resolve_otm_rungs(prof, default=3)

    print(
        f"1s TP/SL scan {args.start_date}..{args.end_date} "
        f"tp={tps} sl={sls} max_hold={args.max_hold_sec}s rules={want_rules}",
        flush=True,
    )

    # Collect unique (session, rule, dir, entry) fills once; score TP/SL grid offline.
    signals: list[dict[str, Any]] = []
    for di, date in enumerate(dates):
        if di % 5 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)}) sig={len(signals)}", flush=True)
        for sym in symbols:
            raw = load_stock_1s_day(stock_1s, sym, date)
            if raw.empty:
                continue
            ts = pd.to_datetime(raw["timestamp"])
            ts = ts.dt.tz_localize(NY) if ts.dt.tz is None else ts.dt.tz_convert(NY)
            raw = raw.copy()
            raw["timestamp"] = ts
            t = raw["timestamp"].dt.time
            day = raw[(t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("16:00").time())]
            if day.empty:
                continue
            arr = prepare_day_arrays(day)
            tday = load_option_trades(trades_root, sym, date)
            if tday is None or tday.empty:
                continue
            trade_paths = _paths_by_ticker(tday)
            if not trade_paths:
                continue
            by_dte = multi_idx.get((sym, date))
            for sess_name, s0, s1 in active:
                t_start = pd.Timestamp(f"{date} {s0}:00", tz=NY)
                t_end = pd.Timestamp(f"{date} {s1}:00", tz=NY)
                tcur = t_start + pd.Timedelta(seconds=max(120, int(args.stride_sec)))
                stride = pd.Timedelta(seconds=int(args.stride_sec))
                while tcur < t_end:
                    feat = features_at(arr, tcur)
                    if feat is None:
                        tcur += stride
                        continue
                    fired: dict[str, str] = {}
                    for rn in want_rules:
                        d, _ = RULES[rn](feat)
                        if d in ("UP", "DN"):
                            fired[rn] = d
                    if not fired:
                        tcur += stride
                        continue
                    for direction in sorted(set(fired.values())):
                        ticker, dte, _ = resolve_open_lock_contract(
                            by_dte,
                            direction=direction,
                            moneyness="ATM",
                            spot=float(feat["px"]),
                            prefer_dte=0,
                            allowed_dte=[0, 1, 2],
                            clear_otm_thresh=0.01,
                            ladder=True,
                            otm_rungs=otm_rungs,
                        )
                        if not ticker:
                            continue
                        key = str(ticker).replace("O:", "")
                        path = trade_paths.get(key)
                        if path is None:
                            continue
                        pts, plast = path
                        rules_here = sorted(r for r, d in fired.items() if d == direction)
                        signals.append(
                            {
                                "date": date,
                                "symbol": sym,
                                "session": sess_name,
                                "dir": direction,
                                "entry_ts": str(tcur),
                                "ticker": ticker,
                                "dte": dte,
                                "rules": rules_here,
                                "pts": pts,
                                "plast": plast,
                            }
                        )
                    tcur += stride

    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"signals={len(signals)} scoring TP/SL grid…", flush=True)

    score_rows = []
    best_trades: dict[str, pd.DataFrame] = {}
    for tp in tps:
        for sl in sls:
            # per session × rule
            for sess_name, _, _ in active:
                for rn in want_rules:
                    raw = []
                    for s in signals:
                        if s["session"] != sess_name or rn not in s["rules"]:
                            continue
                        sim = simulate_trade_tpsl(
                            s["pts"],
                            s["plast"],
                            to_ny(s["entry_ts"]),
                            tp=tp,
                            sl=sl,
                            max_hold_sec=int(args.max_hold_sec),
                            slip=float(args.slip),
                        )
                        if sim is None or not np.isfinite(sim["ret"]):
                            continue
                        et = to_ny(s["entry_ts"])
                        raw.append(
                            {
                                "date": s["date"],
                                "symbol": s["symbol"],
                                "dir": s["dir"],
                                "entry_ts": str(et),
                                "exit_ts": str(et + pd.Timedelta(seconds=sim["hold_sec"])),
                                "ret": sim["ret"],
                                "exit_reason": sim["reason"],
                                "hold_sec": sim["hold_sec"],
                            }
                        )
                    st = _port(raw)
                    row = {
                        "session": sess_name,
                        "rule": rn,
                        "tp": tp,
                        "sl": sl,
                        "max_hold_sec": int(args.max_hold_sec),
                        "n_signals": int(len(raw)),
                        **st,
                    }
                    score_rows.append(row)
                    key = f"{sess_name}|{rn}|tp{tp}|sl{sl}"
                    if st.get("n", 0) > 0 and st.get("mean") is not None and st["mean"] > 0:
                        best_trades[key] = pd.DataFrame(raw)
                    print(
                        f"[{sess_name} {rn} tp={tp} sl={sl}] n={st['n']} mean={st['mean']} "
                        f"add={st['add']:+.3f} day_win={st['day_win']} "
                        f"tp%={st.get('frac_tp')} sl%={st.get('frac_sl')} mh%={st.get('frac_max_hold')}",
                        flush=True,
                    )

    score = pd.DataFrame(score_rows)
    score.to_csv(out / "scoreboard.csv", index=False)
    picks = []
    if len(score):
        ok = score[
            (score["mean"].fillna(-1) > 0)
            & (score["add"].fillna(0) > 0)
            & (score["day_win"].fillna(0) >= 0.55)
            & (score["n"].fillna(0) >= 30)
            & (score["frac_max_hold"].fillna(1) <= 0.50)  # majority exit via TP/SL
        ].sort_values(["session", "add"], ascending=[True, False])
        picks = ok.to_dict(orient="records")
        for i, p in enumerate(picks[:10]):
            key = f"{p['session']}|{p['rule']}|tp{p['tp']}|sl{p['sl']}"
            if key in best_trades:
                best_trades[key].to_csv(out / f"trades_pick{i}_{p['rule']}_tp{p['tp']}_sl{p['sl']}.csv", index=False)

    summary = {
        "start": args.start_date,
        "end": args.end_date,
        "exit": "tp_sl_first_passage",
        "max_hold_sec_safety": int(args.max_hold_sec),
        "tps": tps,
        "sls": sls,
        "rules": want_rules,
        "n_signals_raw": int(len(signals)),
        "n_picks": int(len(picks)),
        "picks": picks[:30],
        "note": (
            "No fixed clock hold as primary exit. TP/SL on option trade last ± slip. "
            "max_hold is safety flatten only; pick gate requires frac_max_hold<=0.5."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "picks.json").write_text(json.dumps(picks[:30], indent=2, default=str), encoding="utf-8")
    print(f"\n=== picks ({len(picks)}) ===", flush=True)
    print(json.dumps(picks[:15], indent=2, default=str), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
