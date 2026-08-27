#!/usr/bin/env python3
"""Make AM pocket champ executable: delayed / quote-ready entry dual.

Champ freezes on trade-last at signal ts, but 09:30 NBBO coverage fails FillSpec.
This scan asks two AM-strategy questions:

  1) Trade-path: if we delay entry 30–300s after the pocket signal, does dual
     survive? (If delay kills edge, quote-wait is hopeless without new data.)
  2) Quote-path: wait up to W seconds for first good NBBO, then TP8/SL15/h240
     FillSpec — can coverage + dual pass?

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_pocket_quote_ready \\
    --tag research_am_pocket_quote_ready
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

from maga7.common.config import load_profile
from maga7.common.fills import FillSpec
from maga7.common.option_quote_tpsl import simulate_quote_tpsl, spread_pct
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import load_quotes, path_for_ticker, to_ny
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_delayed_confirm_quote_dual import _prep_path
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
EXIT = {"mode": "tpsl", "tp": 0.08, "sl": 0.15, "max_hold": 240}
NY = "America/New_York"


def _window_of(date: str) -> str | None:
    for name, a, b in WINDOWS:
        if a <= date <= b:
            return name
    return None


def entry_quote_wait(
    path: pd.DataFrame,
    signal_ts: pd.Timestamp,
    *,
    max_wait_sec: float,
    max_spread_pct: float,
    min_mid: float,
) -> dict[str, Any] | None:
    """First usable quote within [signal, signal+max_wait] (intentional delay)."""
    if path is None or path.empty:
        return None
    t0 = to_ny(signal_ts)
    t1 = t0 + pd.Timedelta(seconds=float(max_wait_sec))
    after = path[(path["timestamp"] >= t0) & (path["timestamp"] <= t1)]
    if after.empty:
        return None
    for i in range(len(after)):
        r = after.iloc[i]
        ts = to_ny(r["timestamp"])
        bid, ask = float(r["bid"]), float(r["ask"])
        if not (np.isfinite(bid) and np.isfinite(ask) and ask > bid > 0):
            continue
        mid = 0.5 * (bid + ask)
        if mid < float(min_mid):
            continue
        sp = spread_pct(bid, ask)
        if sp > float(max_spread_pct):
            continue
        # path for TP/SL must start at accepted quote
        rest = path[path["timestamp"] >= ts]
        return {
            "entry_ts": ts,
            "bid": bid,
            "ask": ask,
            "mid": float(mid),
            "spread_pct": float(sp),
            "lag_sec": float((ts - t0).total_seconds()),
            "after": rest,
        }
    return None


def simulate_quote_tpsl_wait(
    path: pd.DataFrame,
    signal_ts: pd.Timestamp,
    *,
    tp: float,
    sl: float,
    max_hold_sec: int,
    fill: FillSpec,
    max_wait_sec: float,
    max_spread_pct: float,
    min_mid: float,
) -> dict[str, Any] | None:
    ent = entry_quote_wait(
        path,
        signal_ts,
        max_wait_sec=max_wait_sec,
        max_spread_pct=max_spread_pct,
        min_mid=min_mid,
    )
    if ent is None:
        return None
    # reuse simulate from accepted entry by shifting signal to entry_ts with tiny lag
    sim = simulate_quote_tpsl(
        path,
        ent["entry_ts"],
        tp=tp,
        sl=sl,
        max_hold_sec=max_hold_sec,
        fill=fill,
        max_lag_sec=2.0,
        max_spread_pct=max_spread_pct,
        min_mid=min_mid,
    )
    if sim is None:
        return None
    return {**sim, "signal_lag_sec": float(ent["lag_sec"]), "wait_entry": True}


def _score_raw(
    raw: list[dict[str, Any]],
    *,
    position_frac: float,
    max_concurrent: int,
    cooldown: float,
) -> dict[str, Any]:
    disc = [t for t in raw if t["window"] == "may_jul09"]
    blind = [t for t in raw if t["window"] == "jul10_23"]
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
    disc_c = float(st_d.get("compound") or 0)
    blind_c = float(st_b.get("compound") or 0)
    row: dict[str, Any] = {
        "n_raw": len(raw),
        "n_disc": len(disc),
        "n_blind": len(blind),
        "may": months.get("2026-05"),
        "jun": months.get("2026-06"),
        "jul": months.get("2026-07"),
        # Pocket research dual: both windows compound > 0 with enough fills
        "dual_pass": bool(
            len(disc) >= 8 and len(blind) >= 3 and disc_c > 0 and blind_c > 0
        ),
    }
    for k, v in st_d.items():
        row[f"disc_{k}"] = v
    for k, v in st_b.items():
        row[f"blind_{k}"] = v
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--enriched", default=str(DEFAULT_ENRICHED))
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--tag", default="research_am_pocket_quote_ready")
    ap.add_argument("--entry", default="vd+cont60+mf100+volr12")
    ap.add_argument("--trade-delays", default="0,30,60,120,180,300")
    ap.add_argument("--quote-waits", default="5,30,60,120,300,600,900,1800")
    ap.add_argument("--max-spread", type=float, default=0.15)
    ap.add_argument("--min-mid", type=float, default=0.05)
    ap.add_argument("--entry-frac", type=float, default=0.75)
    ap.add_argument("--exit-frac", type=float, default=0.75)
    ap.add_argument("--position-frac", type=float, default=0.20)
    ap.add_argument("--max-concurrent", type=int, default=5)
    ap.add_argument("--cooldown-minutes", type=float, default=10.0)
    ap.add_argument("--slip", type=float, default=0.01)
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    out = Path(prof["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    quote_root = Path(prof["_paths"]["quote_1s_root"])
    trades_root = Path(args.trades_root)
    fill = FillSpec(entry_frac=float(args.entry_frac), exit_frac=float(args.exit_frac))
    delays = [int(x) for x in args.trade_delays.split(",") if x.strip()]
    waits = [int(x) for x in args.quote_waits.split(",") if x.strip()]

    probes = pd.read_csv(args.enriched)
    probes["entry_ts"] = pd.to_datetime(probes["entry_ts"])
    probes = probes[probes["enrich_ok"] == True].copy()  # noqa: E712
    pdf = pd.DataFrame(sorted(POCKET_SETS["no_b_up"]), columns=["session", "tod_bucket", "dir"])
    probes = probes.merge(pdf, on=["session", "tod_bucket", "dir"], how="inner")
    gate = dict(build_gates())[str(args.entry)]
    probes = probes[probes.apply(gate, axis=1)].copy()
    print(f"champ entry={args.entry} n={len(probes)}", flush=True)

    # --- trade paths ---
    tcache: dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    def tpaths(date: str, sym: str):
        key = (date, sym)
        if key not in tcache:
            tday = load_option_trades(trades_root, sym, date)
            tcache[key] = (
                _paths_by_ticker(tday) if tday is not None and not tday.empty else {}
            )
        return tcache[key]

    trade_prep: list[dict[str, Any]] = []
    for _, r in probes.iterrows():
        date, sym = str(r["date"]), str(r["symbol"])
        w = _window_of(date)
        if w is None:
            continue
        arr = tpaths(date, sym).get(str(r["ticker"]).replace("O:", ""))
        if arr is None:
            continue
        trade_prep.append(
            {
                "date": date,
                "symbol": sym,
                "dir": str(r["dir"]),
                "session": str(r["session"]),
                "window": w,
                "signal_ts": to_ny(pd.Timestamp(r["entry_ts"])),
                "ts_ns": arr[0],
                "last": arr[1],
                "oracle_ret": float(r["oracle_ret"]),
            }
        )
    print(f"trade_prep={len(trade_prep)}", flush=True)

    # --- quote paths ---
    qcache: dict[tuple[str, str], pd.DataFrame | None] = {}

    def qday(date: str, sym: str) -> pd.DataFrame | None:
        key = (date, sym)
        if key not in qcache:
            qcache[key] = _prep_path(load_quotes(quote_root, sym, date))
        return qcache[key]

    quote_prep: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for _, r in probes.iterrows():
        date, sym = str(r["date"]), str(r["symbol"])
        w = _window_of(date)
        if w is None:
            continue
        day_q = qday(date, sym)
        ticker = str(r["ticker"]).replace("O:", "")
        pp = _prep_path(path_for_ticker(day_q, ticker)) if day_q is not None else None
        et = to_ny(pd.Timestamp(r["entry_ts"]))
        first_lag = None
        morn_n = 0
        if pp is not None and not pp.empty:
            after = pp[pp["timestamp"] >= et]
            if not after.empty:
                first_lag = float((to_ny(after.iloc[0]["timestamp"]) - et).total_seconds())
            t = pp["timestamp"]
            morn_n = int(((t >= et) & (t < et + pd.Timedelta(minutes=30))).sum())
        coverage_rows.append(
            {
                "date": date,
                "symbol": sym,
                "ticker": ticker,
                "dir": str(r["dir"]),
                "window": w,
                "has_path": pp is not None and not pp.empty,
                "first_quote_lag_sec": first_lag,
                "quotes_in_30m": morn_n,
            }
        )
        if pp is None or pp.empty:
            continue
        quote_prep.append(
            {
                "date": date,
                "symbol": sym,
                "dir": str(r["dir"]),
                "session": str(r["session"]),
                "window": w,
                "signal_ts": et,
                "path": pp,
            }
        )
    cov = pd.DataFrame(coverage_rows)
    cov.to_csv(out / "quote_coverage.csv", index=False)
    print(
        f"quote_prep={len(quote_prep)} "
        f"median_first_lag={cov['first_quote_lag_sec'].median()} "
        f"frac_lag<=60={(cov['first_quote_lag_sec'] <= 60).mean() if len(cov) else 0}",
        flush=True,
    )

    score_rows: list[dict[str, Any]] = []

    # Trade delayed cells
    for dly in delays:
        raw: list[dict[str, Any]] = []
        for p in trade_prep:
            et = p["signal_ts"] + pd.Timedelta(seconds=int(dly))
            win = _path_window(
                p["ts_ns"],
                p["last"],
                et,
                max_hold_sec=900,
                slip=float(args.slip),
            )
            if win is None:
                continue
            rets, holds, _, _ = win
            m = holds <= EXIT["max_hold"] + 1e-9
            rets, holds = rets[m], holds[m]
            if len(rets) < 2:
                continue
            sim = simulate_exit(rets, holds, mode=EXIT["mode"], params=EXIT)
            if not np.isfinite(sim["ret"]):
                continue
            raw.append(
                {
                    "date": p["date"],
                    "symbol": p["symbol"],
                    "dir": p["dir"],
                    "session": p["session"],
                    "window": p["window"],
                    "entry_ts": et,
                    "exit_ts": et + pd.Timedelta(seconds=float(sim["hold_sec"])),
                    "ret": float(sim["ret"]),
                    "exit_reason": str(sim["reason"]),
                    "hold_sec": float(sim["hold_sec"]),
                    "oracle_ret": float(p["oracle_ret"]),
                }
            )
        cell = _score_raw(
            raw,
            position_frac=float(args.position_frac),
            max_concurrent=int(args.max_concurrent),
            cooldown=float(args.cooldown_minutes),
        )
        cell.update({"family": "trade_delay", "param": dly, "gate": f"trade_d{dly}"})
        score_rows.append(cell)
        print(
            f"  trade_d{dly}: n={cell['n_raw']} disc={cell.get('disc_compound')} "
            f"blind={cell.get('blind_compound')} dual={cell['dual_pass']}",
            flush=True,
        )

    # Quote wait cells
    for wsec in waits:
        raw = []
        n_rej = 0
        lags: list[float] = []
        for p in quote_prep:
            sim = simulate_quote_tpsl_wait(
                p["path"],
                p["signal_ts"],
                tp=float(EXIT["tp"]),
                sl=float(EXIT["sl"]),
                max_hold_sec=int(EXIT["max_hold"]),
                fill=fill,
                max_wait_sec=float(wsec),
                max_spread_pct=float(args.max_spread),
                min_mid=float(args.min_mid),
            )
            if sim is None:
                n_rej += 1
                continue
            lags.append(float(sim.get("signal_lag_sec") or np.nan))
            et = sim["entry_ts"]
            raw.append(
                {
                    "date": p["date"],
                    "symbol": p["symbol"],
                    "dir": p["dir"],
                    "session": p["session"],
                    "window": p["window"],
                    "entry_ts": et,
                    "exit_ts": sim["exit_ts"],
                    "ret": float(sim["ret"]),
                    "exit_reason": str(sim["reason"]),
                    "hold_sec": float(sim["hold_sec"]),
                    "oracle_ret": float("nan"),
                }
            )
        cell = _score_raw(
            raw,
            position_frac=float(args.position_frac),
            max_concurrent=int(args.max_concurrent),
            cooldown=float(args.cooldown_minutes),
        )
        cell.update(
            {
                "family": "quote_wait",
                "param": wsec,
                "gate": f"quote_w{wsec}",
                "n_reject": n_rej,
                "fill_rate": (len(raw) / max(len(quote_prep), 1)),
                "median_signal_lag": float(np.nanmedian(lags)) if lags else None,
            }
        )
        score_rows.append(cell)
        print(
            f"  quote_w{wsec}: n={cell['n_raw']} rej={n_rej} fill={cell['fill_rate']:.2f} "
            f"lag_p50={cell['median_signal_lag']} disc={cell.get('disc_compound')} "
            f"blind={cell.get('blind_compound')} dual={cell['dual_pass']}",
            flush=True,
        )

    sb = pd.DataFrame(score_rows)
    sb.to_csv(out / "scoreboard.csv", index=False)

    trade_ok = sb[(sb["family"] == "trade_delay") & (sb["dual_pass"])]
    quote_ok = sb[(sb["family"] == "quote_wait") & (sb["dual_pass"]) & (sb["fill_rate"] >= 0.5)]
    summary = {
        "protocol": "pocket_champ_delayed_and_quote_ready",
        "entry": args.entry,
        "exit": EXIT,
        "n_champ": len(probes),
        "coverage": {
            "n": int(len(cov)),
            "has_path_frac": float(cov["has_path"].mean()) if len(cov) else 0,
            "median_first_lag_sec": (
                float(cov["first_quote_lag_sec"].median())
                if cov["first_quote_lag_sec"].notna().any()
                else None
            ),
            "frac_first_lag_le_60": float((cov["first_quote_lag_sec"] <= 60).mean())
            if cov["first_quote_lag_sec"].notna().any()
            else 0,
            "frac_first_lag_le_300": float((cov["first_quote_lag_sec"] <= 300).mean())
            if cov["first_quote_lag_sec"].notna().any()
            else 0,
        },
        "trade_dual_pass_delays": [int(x) for x in trade_ok["param"].tolist()],
        "quote_dual_pass_waits": [int(x) for x in quote_ok["param"].tolist()],
        "verdict": (
            "QUOTE_READY_PASS"
            if len(quote_ok)
            else (
                "TRADE_DELAY_OK_QUOTE_FAIL"
                if len(trade_ok)
                else "FAIL"
            )
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
