#!/usr/bin/env python3
"""AM sleeve A+B stream vs offline scout parity (recent week).

Compares Mag7Scanner live-path am_pulse / am_pulse_extension scout alerts
against offline ``scan_day`` for the same profile windows.

PASS = scout keys match within --tol-sec (contract skip is reported soft).

Usage:
  python -m maga7.tools.run_am_ab_stream_parity \\
    --start 2026-07-20 --end 2026-07-24
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.am_pulse_scout import (
    load_am_pulse_lane_cfg,
    scan_day,
    scout_config_from_live,
)
from maga7.common.config import load_profile
from maga7.common.replay import month_list
from maga7.common.signals import load_stock_month_files
from maga7.live.scanner import Mag7Scanner, ScannerSignal

FREEZE = (
    ROOT
    / "maga7"
    / "CONFIG"
    / "strategy_profiles"
    / "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
OUT_DEFAULT = Path("/mnt/s990/data/maga7/results/am_ab_stream_parity_week")


def _day_frame(sdf: pd.DataFrame, date: str) -> pd.DataFrame:
    if sdf is None or sdf.empty:
        return pd.DataFrame()
    d = sdf[sdf["date"].astype(str) == str(date)].copy()
    if d.empty:
        return d
    return d.sort_values("timestamp")


def _offline_lane(
    stock_by: dict[str, pd.DataFrame],
    date: str,
    live_cfg: dict[str, Any],
    route: str,
) -> pd.DataFrame:
    if not bool(live_cfg.get("enabled", False)):
        return pd.DataFrame()
    cfg = scout_config_from_live(live_cfg)
    rows: list[dict[str, Any]] = []
    for sym, sdf in stock_by.items():
        day = _day_frame(sdf, date)
        if day.empty:
            continue
        for h in scan_day(day, date=date, symbol=sym, cfg=cfg):
            rows.append(
                {
                    "date": date,
                    "symbol": str(h.symbol).upper(),
                    "dir": str(h.dir).upper(),
                    "route": route,
                    "arm": str(h.arm),
                    "entry_ts": pd.Timestamp(h.ts),
                    "fav_from_open": float(h.fav_from_open),
                    "day_open": float(h.day_open),
                    "px": float(h.px),
                }
            )
    return pd.DataFrame(rows)


def _stream_day(
    profile: dict[str, Any],
    stock_by: dict[str, pd.DataFrame],
    date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (scout_alerts_reaching_emit, emitted_signals)."""
    sc = Mag7Scanner.from_profile(profile)
    scout_rows: list[dict[str, Any]] = []
    emit_rows: list[dict[str, Any]] = []

    _orig_emit = sc._emit_am_pulse_lane

    def _wrap_emit(lane: str, alert: Any) -> ScannerSignal | None:
        route = "am_pulse_extension" if lane == "am_pulse_extension" else "am_pulse"
        row = {
            "date": str(alert.date),
            "symbol": str(alert.symbol).upper(),
            "dir": str(alert.dir).upper(),
            "route": route,
            "arm": str(alert.arm),
            "entry_ts": pd.Timestamp(alert.ts),
            "fav_from_open": float(alert.fav_from_open),
            "day_open": float(alert.day_open),
            "px": float(alert.px),
            "emitted": False,
            "contract": "",
            "dte": None,
            "skip_reason": "",
        }
        sig = _orig_emit(lane, alert)
        if sig is None:
            # Infer common skip: no contract left pending empty; blackout counted in skip.
            row["skip_reason"] = "emit_none"
        else:
            row["emitted"] = True
            row["contract"] = str(sig.contract or "")
            row["dte"] = (sig.meta or {}).get("sig_dte")
            emit_rows.append(
                {
                    "date": sig.date,
                    "symbol": str(sig.symbol).upper(),
                    "dir": str(sig.direction).upper(),
                    "route": str((sig.meta or {}).get("route") or route),
                    "entry_ts": pd.Timestamp(sig.sig_ts),
                    "fav_from_open": float((sig.meta or {}).get("fav_from_open") or 0.0),
                    "day_open": float(alert.day_open),
                    "contract": str(sig.contract or ""),
                    "dte": (sig.meta or {}).get("sig_dte"),
                    "option_type": "",
                    "confirm_abort": bool((sig.meta or {}).get("confirm_abort")),
                }
            )
        scout_rows.append(row)
        return sig

    sc._emit_am_pulse_lane = _wrap_emit  # type: ignore[method-assign]

    events: list[tuple[pd.Timestamp, str, dict[str, Any]]] = []
    for sym, sdf in stock_by.items():
        day = _day_frame(sdf, date)
        if day.empty:
            continue
        for r in day.itertuples(index=False):
            ts = pd.Timestamp(r.timestamp)
            events.append(
                (
                    ts,
                    sym,
                    {
                        "timestamp": ts,
                        "open": float(r.open),
                        "high": float(r.high),
                        "low": float(r.low),
                        "close": float(r.close),
                        "volume": float(getattr(r, "volume", 0.0) or 0.0),
                    },
                )
            )
    events.sort(key=lambda x: (x[0], x[1]))
    for _ts, sym, bar in events:
        sc.on_stock_bar(sym, bar)

    return pd.DataFrame(scout_rows), pd.DataFrame(emit_rows)


def _key_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "symbol", "dir", "route", "entry_ts"])
    out = df.copy()
    out["entry_ts"] = pd.to_datetime(out["entry_ts"])
    return out.sort_values(["date", "route", "symbol", "dir", "entry_ts"]).reset_index(drop=True)


def _match(stream: pd.DataFrame, offline: pd.DataFrame, tol_sec: int = 90) -> dict[str, Any]:
    a = _key_df(stream)
    b = _key_df(offline)
    if a.empty and b.empty:
        return {
            "stream_n": 0,
            "offline_n": 0,
            "matched": 0,
            "only_stream": 0,
            "only_offline": 0,
            "ok": True,
            "pairs": [],
            "only_stream_rows": [],
            "only_offline_rows": [],
        }
    used_b: set[int] = set()
    pairs: list[dict[str, Any]] = []
    only_s: list[dict[str, Any]] = []
    for _, r in a.iterrows():
        cand = b[
            (b["date"].astype(str) == str(r["date"]))
            & (b["symbol"] == r["symbol"])
            & (b["dir"] == r["dir"])
            & (b["route"] == r["route"])
            & (~b.index.isin(used_b))
        ]
        best_i = None
        best_dt = None
        for i, c in cand.iterrows():
            dt = abs((pd.Timestamp(c["entry_ts"]) - pd.Timestamp(r["entry_ts"])).total_seconds())
            if dt <= tol_sec and (best_dt is None or dt < best_dt):
                best_i, best_dt = int(i), float(dt)
        if best_i is None:
            only_s.append({k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in r.to_dict().items()})
            continue
        used_b.add(best_i)
        c = b.loc[best_i]
        pairs.append(
            {
                "date": r["date"],
                "symbol": r["symbol"],
                "dir": r["dir"],
                "route": r["route"],
                "stream_ts": str(r["entry_ts"]),
                "offline_ts": str(c["entry_ts"]),
                "dt_sec": best_dt,
                "stream_fo": float(r.get("fav_from_open") or 0.0),
                "offline_fo": float(c.get("fav_from_open") or 0.0),
                "stream_day_open": float(r.get("day_open") or 0.0),
                "offline_day_open": float(c.get("day_open") or 0.0),
                "fo_abs_diff": abs(float(r.get("fav_from_open") or 0.0) - float(c.get("fav_from_open") or 0.0)),
                "day_open_abs_diff": abs(float(r.get("day_open") or 0.0) - float(c.get("day_open") or 0.0)),
                "emitted": bool(r.get("emitted")) if "emitted" in r.index else True,
                "contract": str(r.get("contract") or ""),
                "dte": r.get("dte"),
            }
        )
    only_o = [
        {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in b.loc[i].to_dict().items()}
        for i in b.index
        if int(i) not in used_b
    ]
    ok = len(only_s) == 0 and len(only_o) == 0
    return {
        "stream_n": int(len(a)),
        "offline_n": int(len(b)),
        "matched": int(len(pairs)),
        "only_stream": int(len(only_s)),
        "only_offline": int(len(only_o)),
        "ok": bool(ok),
        "pairs": pairs,
        "only_stream_rows": only_s,
        "only_offline_rows": only_o,
    }


def _b_checks(emit: pd.DataFrame, profile: dict[str, Any]) -> dict[str, Any]:
    """B lane: dte==0 when present; confirm_abort only_dirs UP on profile."""
    amx = dict(profile.get("am_pulse_extension") or {})
    ca = dict(amx.get("confirm_abort") or {})
    only_dirs = [str(x).upper() for x in (ca.get("only_dirs") or [])]
    if emit is None or emit.empty:
        return {
            "n": 0,
            "dte_bad": 0,
            "ca_cfg_ok": sorted(only_dirs) == ["UP"] and bool(ca.get("enabled")),
            "ok": True,
            "rows": [],
        }
    b = emit[emit["route"] == "am_pulse_extension"].copy()
    bad = []
    for _, r in b.iterrows():
        dte = r.get("dte")
        if dte is None or (isinstance(dte, float) and pd.isna(dte)):
            continue
        if int(dte) != 0:
            bad.append({k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in r.to_dict().items()})
    ca_ok = bool(ca.get("enabled")) and sorted(only_dirs) == ["UP"]
    return {
        "n": int(len(b)),
        "dte_bad": int(len(bad)),
        "ca_cfg_ok": ca_ok,
        "ok": len(bad) == 0 and ca_ok,
        "rows": bad,
    }


def _a_fo_cap(pairs: list[dict[str, Any]], cap: float = 0.015) -> dict[str, Any]:
    a_pairs = [p for p in pairs if p.get("route") == "am_pulse"]
    bad = [p for p in a_pairs if float(p.get("stream_fo") or 0.0) > cap + 1e-9]
    return {"n": len(a_pairs), "over_cap": len(bad), "ok": len(bad) == 0, "rows": bad}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", type=Path, default=FREEZE)
    ap.add_argument("--start", default="2026-07-20")
    ap.add_argument("--end", default="2026-07-24")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--tol-sec", type=int, default=90)
    args = ap.parse_args()

    profile = load_profile(args.profile)
    paths = profile.get("_paths") or {}
    stock_root = Path(paths.get("stock_root") or profile.get("paths", {}).get("stock_root"))
    am = load_am_pulse_lane_cfg(profile, "am_pulse")
    amx = load_am_pulse_lane_cfg(profile, "am_pulse_extension")
    symbols = [str(s).upper() for s in profile.get("symbols") or []]
    months = month_list(args.start, args.end)
    print(f"[load] stock_root={stock_root} months={months} symbols={len(symbols)}", flush=True)
    stock_by: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        sdf = load_stock_month_files(stock_root, sym, months)
        if not sdf.empty:
            stock_by[sym] = sdf
    print(f"[load] symbols_with_bars={len(stock_by)}", flush=True)

    dates = pd.bdate_range(args.start, args.end).strftime("%Y-%m-%d").tolist()
    day_reports: list[dict[str, Any]] = []
    all_scout: list[pd.DataFrame] = []
    all_offline: list[pd.DataFrame] = []
    all_emit: list[pd.DataFrame] = []

    for date in dates:
        if not any(not _day_frame(sdf, date).empty for sdf in stock_by.values()):
            print(f"[skip] {date} no bars", flush=True)
            continue
        off_a = _offline_lane(stock_by, date, am, "am_pulse")
        off_b = _offline_lane(stock_by, date, amx, "am_pulse_extension")
        offline = (
            pd.concat([off_a, off_b], ignore_index=True)
            if (not off_a.empty or not off_b.empty)
            else pd.DataFrame()
        )
        scout, emit = _stream_day(profile, stock_by, date)
        cmp_ = _match(scout, offline, tol_sec=int(args.tol_sec))
        bchk = _b_checks(emit, profile)
        a_fo = _a_fo_cap(cmp_["pairs"], cap=float(am.get("max_fav_from_open") or 0.015))
        day_ok = bool(cmp_["ok"] and bchk["ok"] and a_fo["ok"])
        n_emit = int(len(emit)) if emit is not None and not emit.empty else 0
        n_skip = int((~scout["emitted"]).sum()) if scout is not None and not scout.empty else 0
        rep = {
            "date": date,
            "ok": day_ok,
            "compare": {k: v for k, v in cmp_.items() if k not in ("pairs", "only_stream_rows", "only_offline_rows")},
            "emitted_n": n_emit,
            "emit_skip_n": n_skip,
            "b_checks": {k: v for k, v in bchk.items() if k != "rows"},
            "a_fo_cap": {k: v for k, v in a_fo.items() if k != "rows"},
            "pairs": cmp_["pairs"],
            "only_stream": cmp_["only_stream_rows"],
            "only_offline": cmp_["only_offline_rows"],
            "b_dte_bad": bchk["rows"],
            "a_fo_over": a_fo["rows"],
        }
        day_reports.append(rep)
        if scout is not None and not scout.empty:
            all_scout.append(scout)
        if not offline.empty:
            all_offline.append(offline)
        if emit is not None and not emit.empty:
            all_emit.append(emit)
        tag = "PASS" if day_ok else "FAIL"
        print(
            f"[{tag}] {date} stream_scout={cmp_['stream_n']} offline={cmp_['offline_n']} "
            f"match={cmp_['matched']} only_s={cmp_['only_stream']} only_o={cmp_['only_offline']} "
            f"emitted={n_emit} emit_skip={n_skip} b_dte_bad={bchk['dte_bad']} a_fo_over={a_fo['over_cap']}",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    scout_df = pd.concat(all_scout, ignore_index=True) if all_scout else pd.DataFrame()
    offline_df = pd.concat(all_offline, ignore_index=True) if all_offline else pd.DataFrame()
    emit_df = pd.concat(all_emit, ignore_index=True) if all_emit else pd.DataFrame()
    scout_df.to_csv(args.out / "stream_scout_alerts.csv", index=False)
    offline_df.to_csv(args.out / "offline_scout_alerts.csv", index=False)
    emit_df.to_csv(args.out / "stream_emitted_signals.csv", index=False)
    (args.out / "day_reports.json").write_text(
        json.dumps(day_reports, indent=2, default=str), encoding="utf-8"
    )

    n_ok = sum(1 for r in day_reports if r["ok"])
    n_day = len(day_reports)
    overall = {
        "start": args.start,
        "end": args.end,
        "profile": str(args.profile),
        "days": n_day,
        "days_pass": n_ok,
        "ok": n_day > 0 and n_ok == n_day,
        "stream_scout_n": int(len(scout_df)),
        "offline_scout_n": int(len(offline_df)),
        "emitted_n": int(len(emit_df)),
        "tol_sec": int(args.tol_sec),
        "out": str(args.out),
    }
    (args.out / "summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(json.dumps(overall, indent=2), flush=True)
    return 0 if overall["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
