#!/usr/bin/env python3
"""Scan causal 1s feature entry rules for AM/MID session sleeves.

Builds feature snapshots on ``/mnt/s990/data/raw_1s/stocks`` (no 1m bars),
prices options on trade last ± slip, then scores:
  1) per-feature quintile → clock_ret (exploratory)
  2) named AND-gate rules → opportunity portfolio (≤2 concurrent)

Feature set mirrors usable ideas from ``feature_merge_option_raw`` /
``slow_feature_qqq_v2`` but in **seconds**: ret_*, volume_ratio_60, vwap_diff,
ret_div_*, range_*, vol_z, mf100/streak, from_open.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_session_1s_feature_entry \\
    --start-date 2026-05-01 --end-date 2026-07-22 \\
    --tag research_session_1s_feat_may_jul
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
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import to_ny
from maga7.common.session_1s_features import features_at, prepare_day_arrays
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_session_horizon_foresight import (
    SESSIONS,
    _bdates,
    _fwd_trade_rets_arr,
    _paths_by_ticker,
    _ts_ns,
)

NY = "America/New_York"
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")
DEFAULT_STOCK_1S = Path("/mnt/s990/data/raw_1s/stocks")
FREEZE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)

RuleFn = Callable[[dict[str, Any]], tuple[str | None, str]]


def _finite(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def _dir_from_ret(feat: dict[str, Any], key: str, min_abs: float) -> str | None:
    if not _finite(feat.get(key)):
        return None
    r = float(feat[key])
    if abs(r) < min_abs:
        return None
    return "UP" if r > 0 else "DN"


# Named causal rules → (dir, reason) or (None, reason)
def rule_mom60(f: dict[str, Any], *, min_abs: float = 0.0005) -> tuple[str | None, str]:
    return _dir_from_ret(f, "ret_60", min_abs), "mom60"


def rule_mom30(f: dict[str, Any], *, min_abs: float = 0.0005) -> tuple[str | None, str]:
    return _dir_from_ret(f, "ret_30", min_abs), "mom30"


def rule_mom60_volr(
    f: dict[str, Any], *, min_abs: float = 0.0005, volr: float = 1.5
) -> tuple[str | None, str]:
    d = _dir_from_ret(f, "ret_60", min_abs)
    if d is None:
        return None, "mom60_volr"
    if not _finite(f.get("volume_ratio_60")) or float(f["volume_ratio_60"]) < volr:
        return None, "mom60_volr"
    return d, "mom60_volr"


def rule_mom60_vwap_align(
    f: dict[str, Any], *, min_abs: float = 0.0005
) -> tuple[str | None, str]:
    d = _dir_from_ret(f, "ret_60", min_abs)
    if d is None or not _finite(f.get("vwap_diff")):
        return None, "mom60_vwap"
    vd = float(f["vwap_diff"])
    if d == "UP" and vd <= 0:
        return None, "mom60_vwap"
    if d == "DN" and vd >= 0:
        return None, "mom60_vwap"
    return d, "mom60_vwap"


def rule_mom60_retdiv(
    f: dict[str, Any], *, min_abs: float = 0.0005
) -> tuple[str | None, str]:
    d = _dir_from_ret(f, "ret_60", min_abs)
    if d is None or not _finite(f.get("ret_div_60")):
        return None, "mom60_retdiv"
    rd = float(f["ret_div_60"])
    if d == "UP" and rd <= 0:
        return None, "mom60_retdiv"
    if d == "DN" and rd >= 0:
        return None, "mom60_retdiv"
    return d, "mom60_retdiv"


def rule_mom60_volz(
    f: dict[str, Any], *, min_abs: float = 0.0005, vz: float = 1.5
) -> tuple[str | None, str]:
    d = _dir_from_ret(f, "ret_60", min_abs)
    if d is None:
        return None, "mom60_volz"
    if not _finite(f.get("vol_z")) or float(f["vol_z"]) < vz:
        return None, "mom60_volz"
    return d, "mom60_volz"


def rule_mom60_range(
    f: dict[str, Any], *, min_abs: float = 0.0005, rmin: float = 0.0015
) -> tuple[str | None, str]:
    d = _dir_from_ret(f, "ret_60", min_abs)
    if d is None:
        return None, "mom60_range"
    if not _finite(f.get("range_60")) or float(f["range_60"]) < rmin:
        return None, "mom60_range"
    return d, "mom60_range"


def rule_mf100(f: dict[str, Any], *, streak: int = 5) -> tuple[str | None, str]:
    if not _finite(f.get("mf100")):
        return None, "mf100"
    mf = float(f["mf100"])
    if mf > 0 and int(f.get("streak_up") or 0) >= streak:
        return "UP", "mf100"
    if mf < 0 and int(f.get("streak_dn") or 0) >= streak:
        return "DN", "mf100"
    return None, "mf100"


def rule_mom60_mf(
    f: dict[str, Any], *, min_abs: float = 0.0005
) -> tuple[str | None, str]:
    d = _dir_from_ret(f, "ret_60", min_abs)
    if d is None or not _finite(f.get("mf100")):
        return None, "mom60_mf"
    mf = float(f["mf100"])
    if d == "UP" and mf <= 0:
        return None, "mom60_mf"
    if d == "DN" and mf >= 0:
        return None, "mom60_mf"
    return d, "mom60_mf"


def rule_mom60_volr_mf(
    f: dict[str, Any], *, min_abs: float = 0.0005, volr: float = 1.5
) -> tuple[str | None, str]:
    d, _ = rule_mom60_volr(f, min_abs=min_abs, volr=volr)
    if d is None:
        return None, "mom60_volr_mf"
    if not _finite(f.get("mf100")):
        return None, "mom60_volr_mf"
    mf = float(f["mf100"])
    if d == "UP" and mf <= 0:
        return None, "mom60_volr_mf"
    if d == "DN" and mf >= 0:
        return None, "mom60_volr_mf"
    return d, "mom60_volr_mf"


def rule_mom60_fo_cap(
    f: dict[str, Any], *, min_abs: float = 0.0005, fo_max: float = 0.035
) -> tuple[str | None, str]:
    d = _dir_from_ret(f, "ret_60", min_abs)
    if d is None or not _finite(f.get("from_open")):
        return None, "mom60_fo"
    fo = float(f["from_open"])
    if d == "UP" and fo > fo_max:
        return None, "mom60_fo"
    if d == "DN" and fo < -fo_max:
        return None, "mom60_fo"
    return d, "mom60_fo"


def rule_fade_vwap(
    f: dict[str, Any], *, min_abs: float = 0.002
) -> tuple[str | None, str]:
    """Fade extension vs session VWAP (mean-revert)."""
    if not _finite(f.get("vwap_diff")):
        return None, "fade_vwap"
    vd = float(f["vwap_diff"])
    if abs(vd) < min_abs:
        return None, "fade_vwap"
    return ("DN" if vd > 0 else "UP"), "fade_vwap"


RULES: dict[str, RuleFn] = {
    "MOM60": rule_mom60,
    "MOM30": rule_mom30,
    "MOM60_VOLR15": lambda f: rule_mom60_volr(f, volr=1.5),
    "MOM60_VOLR20": lambda f: rule_mom60_volr(f, volr=2.0),
    "MOM60_VWAP": rule_mom60_vwap_align,
    "MOM60_RETDIV": rule_mom60_retdiv,
    "MOM60_VOLZ15": lambda f: rule_mom60_volz(f, vz=1.5),
    "MOM60_RANGE15": lambda f: rule_mom60_range(f, rmin=0.0015),
    "MF100_S5": lambda f: rule_mf100(f, streak=5),
    "MF100_S10": lambda f: rule_mf100(f, streak=10),
    "MOM60_MF": rule_mom60_mf,
    "MOM60_VOLR15_MF": lambda f: rule_mom60_volr_mf(f, volr=1.5),
    "MOM60_FO035": lambda f: rule_mom60_fo_cap(f, fo_max=0.035),
    "FADE_VWAP20": lambda f: rule_fade_vwap(f, min_abs=0.002),
    "FADE_VWAP35": lambda f: rule_fade_vwap(f, min_abs=0.0035),
}


def _port_stats(rows: list[dict[str, Any]], *, h: int) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "mean": None,
            "win": None,
            "add": 0.0,
            "day_win": None,
            "red_days": 0,
            "tpd": 0.0,
        }
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(str(r["date"]), []).append(r)
    sized: list[dict] = []
    for d in sorted(by):
        sized.extend(
            _portfolio_day(
                by[d],
                position_frac=0.10,
                max_concurrent=2,
                cooldown_minutes=max(1.0, h / 60.0),
            )
        )
    if not sized:
        return {
            "n": 0,
            "mean": None,
            "win": None,
            "add": 0.0,
            "day_win": None,
            "red_days": 0,
            "tpd": 0.0,
        }
    t = pd.DataFrame(sized)
    t["pnl_frac"] = t["ret"].astype(float) * t["size"].astype(float)
    day = t.groupby("date")["pnl_frac"].sum()
    return {
        "n": int(len(t)),
        "mean": float(t["ret"].mean()),
        "win": float((t["ret"] > 0).mean()),
        "add": float(t["pnl_frac"].sum()),
        "day_win": float((day > 0).mean()),
        "red_days": int((day < 0).sum()),
        "tpd": float(len(t) / max(t["date"].nunique(), 1)),
        "worst_day": float(day.min()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=FREEZE)
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--tag", default="research_session_1s_feat")
    ap.add_argument("--stock-1s-root", default=str(DEFAULT_STOCK_1S))
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--stride-sec", type=int, default=60)
    ap.add_argument("--horizons", default="30,60,90,120")
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--sessions", default="AM_0930_1000,MID_1230_1330")
    ap.add_argument("--rules", default=",".join(RULES.keys()))
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    symbols = list(prof.get("symbols") or [])
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    want_sess = {x.strip() for x in args.sessions.split(",") if x.strip()}
    active = tuple(s for s in SESSIONS if s[0] in want_sess)
    want_rules = [x.strip() for x in args.rules.split(",") if x.strip() in RULES]
    stock_1s = Path(args.stock_1s_root)
    trades_root = Path(args.trades_root)
    dates = _bdates(args.start_date, args.end_date)

    lock_path = Path(paths.get("open_locked_map") or paths.get("locked_map")).expanduser()
    multi_idx = load_multidte_lock_index(lock_path) if lock_path.is_file() else {}
    otm_rungs = resolve_otm_rungs(prof, default=3)
    allowed_dte = [0, 1, 2]

    print(
        f"1s feature scan {args.start_date}..{args.end_date} "
        f"sessions={[s[0] for s in active]} rules={want_rules}",
        flush=True,
    )

    events: list[dict[str, Any]] = []
    n_miss = 0
    for di, date in enumerate(dates):
        if di % 5 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)}) events={len(events)}", flush=True)
        for sym in symbols:
            raw = load_stock_1s_day(stock_1s, sym, date)
            if raw.empty:
                continue
            # RTH only
            ts = pd.to_datetime(raw["timestamp"])
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize(NY)
            else:
                ts = ts.dt.tz_convert(NY)
            raw = raw.copy()
            raw["timestamp"] = ts
            t = raw["timestamp"].dt.time
            day = raw[(t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("16:00").time())]
            if day.empty:
                continue
            arr = prepare_day_arrays(day)
            tday = load_option_trades(trades_root, sym, date)
            if tday is None or tday.empty:
                n_miss += 1
                continue
            trade_paths = _paths_by_ticker(tday)
            if not trade_paths:
                n_miss += 1
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
                    # evaluate all rules; if any fires, price once per dir
                    fired: dict[str, str] = {}
                    for rname in want_rules:
                        d, _ = RULES[rname](feat)
                        if d in ("UP", "DN"):
                            fired[rname] = d
                    if not fired:
                        tcur += stride
                        continue
                    for direction in sorted(set(fired.values())):
                        spot = float(feat["px"])
                        ticker, dte, _src = resolve_open_lock_contract(
                            by_dte,
                            direction=direction,
                            moneyness="ATM",
                            spot=spot,
                            prefer_dte=0,
                            allowed_dte=allowed_dte,
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
                        fwd = _fwd_trade_rets_arr(
                            pts, plast, tcur, horizons, slip=float(args.slip)
                        )
                        if not fwd:
                            continue
                        base = {
                            "date": date,
                            "symbol": sym,
                            "session": sess_name,
                            "dir": direction,
                            "entry_ts": str(tcur),
                            "ticker": ticker,
                            "dte": dte,
                            **{k: feat[k] for k in feat if k != "px"},
                            "rules": ",".join(
                                sorted(r for r, d in fired.items() if d == direction)
                            ),
                        }
                        for fr in fwd:
                            events.append({**base, **fr})
                    tcur += stride

    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    edf = pd.DataFrame(events)
    if edf.empty:
        print("no events", flush=True)
        return 1
    edf.to_parquet(out / "events.parquet", index=False)
    print(f"events={len(edf)} unique_sig≈{edf.drop_duplicates(['date','symbol','entry_ts','dir']).shape[0]}", flush=True)

    # --- feature quintiles (H=60) ---
    feat_cols = [
        "ret_60",
        "volume_ratio_60",
        "vwap_diff",
        "ret_div_60",
        "range_60",
        "vol_z",
        "mf100",
        "from_open",
        "streak_up",
        "streak_dn",
    ]
    qrows = []
    sub60 = edf[edf["horizon_sec"] == 60].drop_duplicates(
        ["date", "symbol", "entry_ts", "dir"]
    )
    for col in feat_cols:
        if col not in sub60.columns:
            continue
        x = pd.to_numeric(sub60[col], errors="coerce")
        try:
            q = pd.qcut(x, 5, duplicates="drop")
        except Exception:
            continue
        g = sub60.groupby(q, observed=False)["clock_ret"].agg(["size", "mean"])
        for bucket, row in g.iterrows():
            qrows.append(
                {
                    "feature": col,
                    "bucket": str(bucket),
                    "n": int(row["size"]),
                    "clock_mean": float(row["mean"]),
                }
            )
    pd.DataFrame(qrows).to_csv(out / "feature_quintiles_h60.csv", index=False)

    # --- rule scoreboard ---
    score_rows = []
    for sess_name, _, _ in active:
        for h in horizons:
            base = edf[
                (edf["session"] == sess_name) & (edf["horizon_sec"].astype(int) == h)
            ]
            if base.empty:
                continue
            for rname in want_rules:
                # signals where this rule is in the fired list for that dir
                mask = base["rules"].astype(str).str.split(",").apply(
                    lambda xs, rn=rname: rn in xs
                )
                sub = base[mask].drop_duplicates(["date", "symbol", "entry_ts", "dir"])
                raw = []
                for r in sub.itertuples():
                    ret = float(r.clock_ret)
                    if not np.isfinite(ret):
                        continue
                    et = to_ny(r.entry_ts)
                    raw.append(
                        {
                            "date": str(r.date),
                            "symbol": str(r.symbol),
                            "dir": str(r.dir),
                            "entry_ts": str(et),
                            "exit_ts": str(et + pd.Timedelta(seconds=h)),
                            "ret": ret,
                        }
                    )
                st = _port_stats(raw, h=h)
                score_rows.append(
                    {
                        "session": sess_name,
                        "horizon_sec": h,
                        "rule": rname,
                        "n_signals": int(len(sub)),
                        **st,
                    }
                )
                print(
                    f"[{sess_name} H{h} {rname}] n={st['n']} mean={st['mean']} "
                    f"add={st['add']:+.3f} day_win={st['day_win']} red={st['red_days']}",
                    flush=True,
                )

    score = pd.DataFrame(score_rows)
    score.to_csv(out / "rule_scoreboard.csv", index=False)

    # picks: mean>0, day_win>=0.55, n>=30, add>0
    picks = []
    if len(score):
        ok = score[
            (score["mean"].fillna(-1) > 0)
            & (score["add"].fillna(0) > 0)
            & (score["day_win"].fillna(0) >= 0.55)
            & (score["n"].fillna(0) >= 30)
        ].sort_values(["session", "add"], ascending=[True, False])
        picks = ok.to_dict(orient="records")

    summary = {
        "start": args.start_date,
        "end": args.end_date,
        "stock_1s_root": str(stock_1s),
        "trades_root": str(trades_root),
        "stride_sec": int(args.stride_sec),
        "horizons": horizons,
        "rules": want_rules,
        "n_events": int(len(edf)),
        "n_miss_trades_days": int(n_miss),
        "n_picks": int(len(picks)),
        "picks": picks[:40],
        "note": (
            "All stock features causal on 1s. No left-label 1m. "
            "Option PnL = trade last ± slip clock. Capacity ≤2 concurrent."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "picks.json").write_text(json.dumps(picks[:40], indent=2, default=str), encoding="utf-8")
    print(f"\n=== picks ({len(picks)}) ===", flush=True)
    print(json.dumps(picks[:20], indent=2, default=str), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
