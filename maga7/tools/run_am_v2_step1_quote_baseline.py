#!/usr/bin/env python3
"""AM v2 Step1: Mag7 AM ATM quote coverage baseline (no signal PnL).

Clock-sample 09:30–11:30 every ``--stride-sec``; resolve open-lock ATM;
probe quote FillSpec entry gates. Reports coverage / lag / spread by window
and TOD bucket.

PASS: artifacts written (this step does not require positive PnL).
FAIL: no usable probes → fix data before Step2.

Example:
  PYTHONPATH=. python -m maga7.tools.run_am_v2_step1_quote_baseline \\
    --tag research_am_v2_step1_quote_baseline
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
from maga7.common.option_quote_tpsl import entry_quote_row, spread_pct
from maga7.common.replay import load_quotes, path_for_ticker, to_ny
from maga7.common.session_1s_features import prepare_day_arrays
from maga7.common.stock_1s import session_dates
from maga7.tools.scan_am_pocket_regime_ladder_v2 import _window_of
from maga7.tools.scan_session_horizon_foresight import _spot_at_arr

PROFILE = "maga7/CONFIG/strategy_profiles/am_v2_executable_path_v1.json"
SPINE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
NY = "America/New_York"
WINDOWS = ("may_jul09", "jul10_23")


def _tod_bucket(ts: pd.Timestamp) -> str:
    hm = ts.hour * 60 + ts.minute
    if hm < 10 * 60:
        return "0930_1000"
    if hm < 10 * 60 + 30:
        return "1000_1030"
    if hm < 11 * 60:
        return "1030_1100"
    return "1100_1130"


def _first_quote_diag(path: pd.DataFrame, entry_ts: pd.Timestamp) -> dict[str, float] | None:
    """Raw first quote at/after t (no gate) for lag/spread diagnostics."""
    if path is None or path.empty:
        return None
    t0 = to_ny(entry_ts)
    after = path[path["timestamp"] >= t0]
    if after.empty:
        return None
    r0 = after.iloc[0]
    ts = to_ny(r0["timestamp"])
    bid, ask = float(r0["bid"]), float(r0["ask"])
    if not (np.isfinite(bid) and np.isfinite(ask) and ask > bid > 0):
        return None
    mid = 0.5 * (bid + ask)
    return {
        "lag_sec": float((ts - t0).total_seconds()),
        "spread_pct": float(spread_pct(bid, ask)),
        "mid": float(mid),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--spine", default=SPINE)
    ap.add_argument("--tag", default="research_am_v2_step1_quote_baseline")
    ap.add_argument("--window-start", default="09:30")
    ap.add_argument("--window-end", default="11:30")
    ap.add_argument("--stride-sec", type=int, default=60)
    ap.add_argument("--max-lag-sec", type=float, default=5.0)
    ap.add_argument("--max-spread-pct", type=float, default=0.15)
    ap.add_argument("--min-mid", type=float, default=0.05)
    ap.add_argument("--start-date", default="2026-05-01")
    ap.add_argument("--end-date", default="2026-07-23")
    ap.add_argument("--max-days", type=int, default=0)
    args = ap.parse_args(argv)

    v2 = load_profile(args.profile)
    spine = load_profile(args.spine)
    paths = spine["_paths"]
    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    stock_1s = Path(paths["stock_1s_root"])
    quote_root = Path(paths["quote_1s_root"])
    lock = load_multidte_lock_index(Path(paths["open_locked_map"]).expanduser())
    otm = resolve_otm_rungs(spine, default=3)
    symbols = list(v2.get("symbols") or spine.get("symbols") or [])

    dates = [
        d
        for d in session_dates(args.start_date, args.end_date)
        if args.start_date <= d <= args.end_date
    ]
    if int(args.max_days) > 0:
        dates = dates[: int(args.max_days)]

    print(
        f"am_v2 step1 quote baseline days={len(dates)} syms={len(symbols)} "
        f"stride={args.stride_sec}s quote_root={quote_root}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    qcache: dict[tuple[str, str], pd.DataFrame | None] = {}

    for di, date in enumerate(dates):
        w = _window_of(date)
        if w is None:
            continue
        if di % 10 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)}) probes={len(rows)}", flush=True)
        for sym in symbols:
            raw = load_stock_1s_day(stock_1s, sym, date)
            if raw is None or raw.empty:
                continue
            sarr = prepare_day_arrays(raw)
            by_dte = lock.get((sym, date))
            if not by_dte:
                continue
            qkey = (sym, date)
            if qkey not in qcache:
                qcache[qkey] = load_quotes(quote_root, sym, date)
            qday = qcache[qkey]
            has_sym_quotes = qday is not None and not qday.empty

            t = to_ny(pd.Timestamp(f"{date} {args.window_start}", tz=NY))
            t1 = to_ny(pd.Timestamp(f"{date} {args.window_end}", tz=NY))
            while t < t1:
                spot = _spot_at_arr(sarr["ts_ns"], sarr["close"], t)
                if spot is None or not np.isfinite(float(spot)):
                    t = t + pd.Timedelta(seconds=int(args.stride_sec))
                    continue
                spot_f = float(spot)
                for direction in ("UP", "DN"):
                    ticker, dte, _ = resolve_open_lock_contract(
                        by_dte,
                        direction=direction,
                        moneyness="ATM",
                        spot=spot_f,
                        prefer_dte=0,
                        allowed_dte=(0, 1, 2),
                        clear_otm_thresh=0.01,
                        ladder=True,
                        otm_rungs=otm,
                    )
                    if not ticker:
                        rows.append(
                            {
                                "date": date,
                                "window": w,
                                "symbol": sym,
                                "dir": direction,
                                "ts": str(t),
                                "tod_bucket": _tod_bucket(t),
                                "ticker": None,
                                "dte": None,
                                "has_contract": False,
                                "has_sym_quotes": has_sym_quotes,
                                "has_ticker_path": False,
                                "has_any_quote": False,
                                "gate_ok": False,
                                "lag_sec": np.nan,
                                "spread_pct": np.nan,
                                "mid": np.nan,
                            }
                        )
                        continue
                    key = str(ticker).replace("O:", "")
                    path = path_for_ticker(qday, key) if has_sym_quotes else None
                    has_path = path is not None and not path.empty
                    diag = _first_quote_diag(path, t) if has_path else None
                    gate = None
                    if has_path:
                        gate = entry_quote_row(
                            path,
                            t,
                            max_lag_sec=float(args.max_lag_sec),
                            max_spread_pct=float(args.max_spread_pct),
                            min_mid=float(args.min_mid),
                        )
                    rows.append(
                        {
                            "date": date,
                            "window": w,
                            "symbol": sym,
                            "dir": direction,
                            "ts": str(t),
                            "tod_bucket": _tod_bucket(t),
                            "ticker": key,
                            "dte": dte,
                            "has_contract": True,
                            "has_sym_quotes": has_sym_quotes,
                            "has_ticker_path": has_path,
                            "has_any_quote": diag is not None,
                            "gate_ok": gate is not None,
                            "lag_sec": diag["lag_sec"] if diag else np.nan,
                            "spread_pct": diag["spread_pct"] if diag else np.nan,
                            "mid": diag["mid"] if diag else np.nan,
                        }
                    )
                t = t + pd.Timedelta(seconds=int(args.stride_sec))

    if not rows:
        print("no probes — FAIL Step1", flush=True)
        (out / "summary.json").write_text(
            json.dumps({"step": 1, "pass": False, "reason": "no_probes"}, indent=2)
        )
        return 1

    df = pd.DataFrame(rows)
    df.to_csv(out / "probes.csv", index=False)

    def _agg(sub: pd.DataFrame) -> dict[str, Any]:
        if sub.empty:
            return {"n": 0}
        any_q = sub["has_any_quote"]
        return {
            "n": int(len(sub)),
            "frac_contract": float(sub["has_contract"].mean()),
            "frac_sym_quotes": float(sub["has_sym_quotes"].mean()),
            "frac_ticker_path": float(sub["has_ticker_path"].mean()),
            "frac_any_quote": float(any_q.mean()),
            "frac_gate_ok": float(sub["gate_ok"].mean()),
            "lag_p50": float(sub.loc[any_q, "lag_sec"].median()) if any_q.any() else None,
            "lag_p90": float(sub.loc[any_q, "lag_sec"].quantile(0.9)) if any_q.any() else None,
            "spread_p50": float(sub.loc[any_q, "spread_pct"].median()) if any_q.any() else None,
        }

    score_rows = []
    for w in WINDOWS + ("ALL",):
        base = df if w == "ALL" else df[df["window"] == w]
        for bucket in ("0930_1000", "1000_1030", "1030_1100", "1100_1130", "ALL"):
            sub = base if bucket == "ALL" else base[base["tod_bucket"] == bucket]
            st = _agg(sub)
            st["window"] = w
            st["tod_bucket"] = bucket
            score_rows.append(st)
    sb = pd.DataFrame(score_rows)
    sb.to_csv(out / "scoreboard.csv", index=False)

    by_sym = (
        df.groupby("symbol")
        .agg(
            n=("gate_ok", "size"),
            frac_gate_ok=("gate_ok", "mean"),
            frac_any_quote=("has_any_quote", "mean"),
            lag_p50=("lag_sec", "median"),
        )
        .reset_index()
    )
    by_sym.to_csv(out / "by_symbol.csv", index=False)

    open_row = sb[(sb.window == "ALL") & (sb.tod_bucket == "0930_1000")]
    all_row = sb[(sb.window == "ALL") & (sb.tod_bucket == "ALL")]
    open_gate = float(open_row.iloc[0]["frac_gate_ok"]) if len(open_row) else 0.0
    all_gate = float(all_row.iloc[0]["frac_gate_ok"]) if len(all_row) else 0.0
    open_any = float(open_row.iloc[0]["frac_any_quote"]) if len(open_row) else 0.0
    data_debt = bool(open_gate < 0.10)

    summary = {
        "protocol": "am_v2_step1_quote_baseline",
        "step": 1,
        "pass": True,
        "north_star": "executable_quote_fill",
        "promotion_mark": "quote_FillSpec",
        "n_probes": int(len(df)),
        "n_days": int(df["date"].nunique()),
        "gate": {
            "max_lag_sec": float(args.max_lag_sec),
            "max_spread_pct": float(args.max_spread_pct),
            "min_mid": float(args.min_mid),
        },
        "frac_gate_ok_all": all_gate,
        "frac_gate_ok_0930_1000": open_gate,
        "frac_any_quote_0930_1000": open_any,
        "data_debt_open_quote": data_debt,
        "next_step": (
            2 if not data_debt else "1b_document_debt_then_step2_with_hard_quote_gates"
        ),
        "scoreboard_head": sb.head(15).to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    def _tbl(x: pd.DataFrame) -> str:
        try:
            return x.to_markdown(index=False)
        except Exception:
            return x.to_string(index=False)

    (out / "REPORT.md").write_text(
        "\n".join(
            [
                "# AM v2 Step1 — Quote baseline",
                "",
                f"- probes: **{len(df)}** · days: **{df['date'].nunique()}**",
                f"- gate_ok ALL: **{all_gate:.1%}** "
                f"(lag≤{args.max_lag_sec}s, spread≤{args.max_spread_pct:.0%})",
                f"- gate_ok 09:30–10:00: **{open_gate:.1%}** "
                f"(any_quote={open_any:.1%})",
                f"- data_debt_open_quote: **{data_debt}**",
                f"- next: **{summary['next_step']}**",
                "",
                "## Scoreboard",
                "",
                _tbl(sb),
                "",
                "## By symbol",
                "",
                _tbl(by_sym),
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("\n=== SCOREBOARD ===", flush=True)
    print(sb.to_string(index=False), flush=True)
    print(
        json.dumps(
            {
                "pass": True,
                "frac_gate_ok_all": all_gate,
                "frac_gate_ok_0930_1000": open_gate,
                "frac_any_quote_0930_1000": open_any,
                "data_debt": data_debt,
                "next": summary["next_step"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
