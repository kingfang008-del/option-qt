#!/usr/bin/env python3
"""CORE DN sync — quote coverage / sync funnel probe (not a promote scan).

Walks the same morph as ``scan_certainty_morph_quote_dual`` champion
(``thr=0.3%``, stock sync 30s, opt sync 30s, CORE 10:30–11:30 DN) and attributes
losses at each gate:

  stock_dn → lock → quote_day → ticker_path → entry_probe
  → stock_sync → opt_sync_quote vs opt_sync_trades → quote_tpsl_fill

Also records raw next-quote lag/spread/mid at DN arms (ungated) so we can see
whether the book is missing, wide, or merely failing FillSpec lookback sync.

Example:
  PYTHONPATH=. python -m maga7.tools.probe_core_dn_sync_quote_coverage \\
    --tag research_core_dn_sync_quote_coverage
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.config import load_profile
from maga7.common.fills import FillSpec
from maga7.common.open_lock import (
    load_multidte_lock_index,
    resolve_open_lock_contract,
    resolve_otm_rungs,
)
from maga7.common.option_quote_tpsl import entry_quote_row, simulate_quote_tpsl, spread_pct
from maga7.common.option_trades import load_option_trades, path_for_ticker_trades
from maga7.common.replay import load_quotes, path_for_ticker, to_ny
from maga7.common.stock_1s import session_dates
from maga7.tools.scan_am_certainty_morph_tpsl import _opt_ret_window, _stock_signed
from maga7.tools.scan_am_delayed_confirm_quote_dual import _prep_path
from maga7.tools.scan_certainty_morph_quote_dual import _quote_opt_ret
from maga7.tools.scan_session_horizon_foresight import (
    _paths_by_ticker,
    _spot_at_arr,
    _stock_arrays,
    _stock_dir_arr,
)

NY = "America/New_York"
PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")
SESS_START, SESS_END = "10:30", "11:30"
WINDOWS = (
    ("may_jul09", "2026-05-01", "2026-07-09"),
    ("jul10_23", "2026-07-10", "2026-07-23"),
)


def _window_of(date: str) -> str | None:
    for wname, a, b in WINDOWS:
        if a <= date <= b:
            return wname
    return None


def _probe_detail(
    path: pd.DataFrame,
    entry_ts: pd.Timestamp,
    *,
    max_lag_sec: float,
    max_spread_pct: float,
    min_mid: float,
) -> dict[str, Any]:
    """Ungated next-quote diagnostics + gated pass/fail reason."""
    out: dict[str, Any] = {
        "has_after": False,
        "lag_sec": None,
        "spread_pct": None,
        "mid": None,
        "bid": None,
        "ask": None,
        "fail": "empty_path",
        "ok": False,
    }
    if path is None or path.empty:
        return out
    t0 = to_ny(entry_ts)
    after = path[path["timestamp"] >= t0]
    if after.empty:
        out["fail"] = "no_quote_after"
        return out
    r0 = after.iloc[0]
    ts = to_ny(r0["timestamp"])
    lag = float((ts - t0).total_seconds())
    bid, ask = float(r0["bid"]), float(r0["ask"])
    out.update({"has_after": True, "lag_sec": lag, "bid": bid, "ask": ask})
    if not (np.isfinite(bid) and np.isfinite(ask) and ask > bid > 0):
        out["fail"] = "bad_book"
        return out
    mid = 0.5 * (bid + ask)
    sp = float(spread_pct(bid, ask))
    out.update({"mid": float(mid), "spread_pct": sp})
    if lag > float(max_lag_sec):
        out["fail"] = "lag"
        return out
    if mid < float(min_mid):
        out["fail"] = "min_mid"
        return out
    if sp > float(max_spread_pct):
        out["fail"] = "spread"
        return out
    out["fail"] = None
    out["ok"] = True
    return out


def _pct(n: int, d: int) -> float | None:
    return float(n / d) if d else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_core_dn_sync_quote_coverage")
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--thr", type=float, default=0.003)
    ap.add_argument("--sync-stock-sec", type=int, default=30)
    ap.add_argument("--sync-opt-sec", type=int, default=30)
    ap.add_argument("--tp", type=float, default=0.20)
    ap.add_argument("--sl", type=float, default=0.15)
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--max-spread-pct", type=float, default=0.15)
    ap.add_argument("--max-lag-sec", type=float, default=3.0)
    ap.add_argument("--min-mid", type=float, default=0.05)
    ap.add_argument("--entry-frac", type=float, default=0.75)
    ap.add_argument("--exit-frac", type=float, default=0.75)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--lookback-sec", type=int, default=60)
    ap.add_argument("--stride-sec", type=int, default=60)
    args = ap.parse_args(argv)

    fill = FillSpec(entry_frac=float(args.entry_frac), exit_frac=float(args.exit_frac))
    prof = load_profile(args.profile)
    paths = prof["_paths"]
    symbols = list(prof.get("symbols") or [])
    stock_1s = Path(paths.get("stock_1s_root") or "/mnt/s990/data/raw_1s/stocks").expanduser()
    quote_root = Path(paths["quote_1s_root"])
    trades_root = Path(args.trades_root).expanduser()
    lock = load_multidte_lock_index(Path(paths["open_locked_map"]).expanduser())
    otm = resolve_otm_rungs(prof, default=3)
    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    start_all = min(w[1] for w in WINDOWS)
    end_all = max(w[2] for w in WINDOWS)
    dates = session_dates(start_all, end_all)
    print(
        f"CORE DN sync quote coverage {start_all}..{end_all} "
        f"thr={args.thr} ss={args.sync_stock_sec} so={args.sync_opt_sec} "
        f"sp≤{args.max_spread_pct} lag≤{args.max_lag_sec}",
        flush=True,
    )

    funnel: dict[str, Counter] = {w[0]: Counter() for w in WINDOWS}
    funnel["ALL"] = Counter()
    rows: list[dict[str, Any]] = []
    probe_fail = Counter()

    for di, date in enumerate(dates):
        wname = _window_of(str(date))
        if wname is None:
            continue
        if di % 10 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)}) rows={len(rows)}", flush=True)
        for sym in symbols:
            day = load_stock_1s_day(stock_1s, sym, date)
            if day is None or day.empty:
                funnel[wname]["no_stock"] += 1
                funnel["ALL"]["no_stock"] += 1
                continue
            ts_ns, px = _stock_arrays(day)
            by_dte = lock.get((sym, date))
            qday = _prep_path(load_quotes(quote_root, sym, date))
            tday = load_option_trades(trades_root, sym, date)
            t_paths = _paths_by_ticker(tday) if tday is not None and not tday.empty else {}

            t0 = pd.Timestamp(f"{date} {SESS_START}:00", tz=NY) + pd.Timedelta(
                seconds=int(args.lookback_sec)
            )
            t1 = pd.Timestamp(f"{date} {SESS_END}:00", tz=NY)
            # Match quote dual: keep scanning until first DN with usable entry probe.
            armed = False
            saw_dn = False
            t = t0
            stride = pd.Timedelta(seconds=int(args.stride_sec))
            while t < t1 and not armed:
                direction, sr = _stock_dir_arr(
                    ts_ns, px, t, int(args.lookback_sec), float(args.thr)
                )
                if direction != "DN":
                    t += stride
                    continue
                if not saw_dn:
                    saw_dn = True
                    for bucket in (wname, "ALL"):
                        funnel[bucket]["stock_dn"] += 1

                if not by_dte:
                    for bucket in (wname, "ALL"):
                        funnel[bucket]["lock_miss"] += 1
                    rows.append(
                        {
                            "date": date,
                            "window": wname,
                            "symbol": sym,
                            "arm_ts": str(to_ny(t)),
                            "stock_ret_lb": float(sr),
                            "stage": "lock_miss",
                            "mode": "first_dn",
                        }
                    )
                    break  # no lock that day/symbol
                for bucket in (wname, "ALL"):
                    funnel[bucket]["lock_ok"] += 1

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
                if not ticker:
                    for bucket in (wname, "ALL"):
                        funnel[bucket]["contract_miss"] += 1
                    # retry later stride (spot may clear rung)
                    t += stride
                    continue
                for bucket in (wname, "ALL"):
                    funnel[bucket]["contract_ok"] += 1

                if qday is None or qday.empty:
                    for bucket in (wname, "ALL"):
                        funnel[bucket]["quote_day_miss"] += 1
                    rows.append(
                        {
                            "date": date,
                            "window": wname,
                            "symbol": sym,
                            "arm_ts": str(to_ny(t)),
                            "ticker": ticker,
                            "dte": dte,
                            "stage": "quote_day_miss",
                        }
                    )
                    break
                for bucket in (wname, "ALL"):
                    funnel[bucket]["quote_day_ok"] += 1

                qpath = _prep_path(path_for_ticker(qday, ticker))
                if qpath is None or qpath.empty:
                    for bucket in (wname, "ALL"):
                        funnel[bucket]["quote_ticker_miss"] += 1
                    t += stride
                    continue
                for bucket in (wname, "ALL"):
                    funnel[bucket]["quote_ticker_ok"] += 1

                detail = _probe_detail(
                    qpath,
                    t,
                    max_lag_sec=float(args.max_lag_sec),
                    max_spread_pct=float(args.max_spread_pct),
                    min_mid=float(args.min_mid),
                )
                if not detail["ok"]:
                    reason = str(detail["fail"] or "probe_fail")
                    probe_fail[reason] += 1
                    for bucket in (wname, "ALL"):
                        funnel[bucket]["entry_probe_fail"] += 1
                        funnel[bucket][f"probe_{reason}"] += 1
                    # retry later DN stride (same as quote dual arm builder)
                    t += stride
                    continue

                armed = True
                for bucket in (wname, "ALL"):
                    funnel[bucket]["entry_probe_ok"] += 1

                # stock sync
                sret = _stock_signed(ts_ns, px, t, int(args.sync_stock_sec), "DN")
                stock_sync = sret is not None and sret >= 0
                if not stock_sync:
                    for bucket in (wname, "ALL"):
                        funnel[bucket]["stock_sync_fail"] += 1
                    rows.append(
                        {
                            "date": date,
                            "window": wname,
                            "symbol": sym,
                            "arm_ts": str(to_ny(t)),
                            "ticker": ticker,
                            "dte": dte,
                            "stage": "stock_sync_fail",
                            "stock_sync_ret": sret,
                            "lag_sec": detail["lag_sec"],
                            "spread_pct": detail["spread_pct"],
                            "mid": detail["mid"],
                        }
                    )
                    break
                for bucket in (wname, "ALL"):
                    funnel[bucket]["stock_sync_ok"] += 1

                # option sync: quote FillSpec lookback vs trades print lookback
                oret_q = _quote_opt_ret(
                    qpath,
                    t,
                    int(args.sync_opt_sec),
                    fill=fill,
                    max_lag_sec=float(args.max_lag_sec),
                    max_spread_pct=float(args.max_spread_pct),
                    min_mid=float(args.min_mid),
                )
                oret_t: float | None = None
                tkey = str(ticker).replace("O:", "").upper()
                tarr = t_paths.get(tkey) or t_paths.get(str(ticker).replace("O:", ""))
                if tarr is not None:
                    oret_t = _opt_ret_window(
                        tarr[0], tarr[1], t, int(args.sync_opt_sec), slip=float(args.slip)
                    )
                else:
                    tp = path_for_ticker_trades(tday, ticker) if tday is not None else None
                    if tp is not None and not tp.empty:
                        ts_t = (
                            pd.to_datetime(tp["timestamp"], utc=True)
                            .astype("int64")
                            .to_numpy()
                        )
                        last_t = tp["last"].astype(float).to_numpy()
                        oret_t = _opt_ret_window(
                            ts_t, last_t, t, int(args.sync_opt_sec), slip=float(args.slip)
                        )

                q_sync = oret_q is not None and oret_q > 0
                t_sync = oret_t is not None and oret_t > 0
                for bucket in (wname, "ALL"):
                    if oret_q is None:
                        funnel[bucket]["opt_quote_none"] += 1
                    elif oret_q <= 0:
                        funnel[bucket]["opt_quote_nonpos"] += 1
                    else:
                        funnel[bucket]["opt_quote_pos"] += 1
                    if oret_t is None:
                        funnel[bucket]["opt_trades_none"] += 1
                    elif oret_t <= 0:
                        funnel[bucket]["opt_trades_nonpos"] += 1
                    else:
                        funnel[bucket]["opt_trades_pos"] += 1
                    if q_sync and t_sync:
                        funnel[bucket]["opt_both_pos"] += 1
                    elif t_sync and not q_sync:
                        funnel[bucket]["opt_trades_only"] += 1
                    elif q_sync and not t_sync:
                        funnel[bucket]["opt_quote_only"] += 1

                if not q_sync:
                    for bucket in (wname, "ALL"):
                        funnel[bucket]["opt_quote_sync_fail"] += 1
                    rows.append(
                        {
                            "date": date,
                            "window": wname,
                            "symbol": sym,
                            "arm_ts": str(to_ny(t)),
                            "ticker": ticker,
                            "dte": dte,
                            "stage": "opt_quote_sync_fail",
                            "opt_ret_quote": oret_q,
                            "opt_ret_trades": oret_t,
                            "trades_would_pass": bool(t_sync),
                            "lag_sec": detail["lag_sec"],
                            "spread_pct": detail["spread_pct"],
                            "mid": detail["mid"],
                        }
                    )
                    break
                for bucket in (wname, "ALL"):
                    funnel[bucket]["opt_quote_sync_ok"] += 1

                sim = simulate_quote_tpsl(
                    qpath,
                    t,
                    tp=float(args.tp),
                    sl=float(args.sl),
                    max_hold_sec=int(args.max_hold_sec),
                    fill=fill,
                    max_lag_sec=float(args.max_lag_sec),
                    max_spread_pct=float(args.max_spread_pct),
                    min_mid=float(args.min_mid),
                )
                if sim is None or not np.isfinite(sim.get("ret", np.nan)):
                    for bucket in (wname, "ALL"):
                        funnel[bucket]["tpsl_fail"] += 1
                    rows.append(
                        {
                            "date": date,
                            "window": wname,
                            "symbol": sym,
                            "arm_ts": str(to_ny(t)),
                            "ticker": ticker,
                            "dte": dte,
                            "stage": "tpsl_fail",
                            "opt_ret_quote": oret_q,
                            "opt_ret_trades": oret_t,
                        }
                    )
                    break
                for bucket in (wname, "ALL"):
                    funnel[bucket]["tpsl_ok"] += 1
                rows.append(
                    {
                        "date": date,
                        "window": wname,
                        "symbol": sym,
                        "arm_ts": str(to_ny(t)),
                        "ticker": ticker,
                        "dte": dte,
                        "stage": "filled",
                        "opt_ret_quote": oret_q,
                        "opt_ret_trades": oret_t,
                        "ret": float(sim["ret"]),
                        "exit_reason": sim["reason"],
                        "hold_sec": sim["hold_sec"],
                        "lag_sec": detail["lag_sec"],
                        "spread_pct": detail["spread_pct"],
                        "mid": detail["mid"],
                    }
                )
                break

    detail_df = pd.DataFrame(rows)
    detail_df.to_csv(out / "arm_funnel_detail.csv", index=False)

    # summary tables
    stages = [
        "stock_dn",
        "lock_ok",
        "contract_ok",
        "quote_day_ok",
        "quote_ticker_ok",
        "entry_probe_ok",
        "stock_sync_ok",
        "opt_quote_sync_ok",
        "tpsl_ok",
    ]
    sum_rows: list[dict[str, Any]] = []
    for wname in [w[0] for w in WINDOWS] + ["ALL"]:
        c = funnel[wname]
        base = int(c.get("stock_dn", 0))
        row: dict[str, Any] = {"window": wname, "stock_dn": base}
        for s in stages[1:]:
            n = int(c.get(s, 0))
            row[s] = n
            row[f"{s}_frac"] = _pct(n, base)
        # attribution extras
        for k in (
            "lock_miss",
            "contract_miss",
            "quote_day_miss",
            "quote_ticker_miss",
            "entry_probe_fail",
            "stock_sync_fail",
            "opt_quote_sync_fail",
            "opt_quote_none",
            "opt_quote_nonpos",
            "opt_quote_pos",
            "opt_trades_none",
            "opt_trades_nonpos",
            "opt_trades_pos",
            "opt_both_pos",
            "opt_trades_only",
            "opt_quote_only",
            "tpsl_fail",
            "probe_no_quote_after",
            "probe_lag",
            "probe_spread",
            "probe_min_mid",
            "probe_bad_book",
        ):
            row[k] = int(c.get(k, 0))
        sum_rows.append(row)

    summary_df = pd.DataFrame(sum_rows)
    summary_df.to_csv(out / "funnel_summary.csv", index=False)

    # among stock_sync_ok: quote vs trades disagreement
    sync_cmp = {}
    if not detail_df.empty and "opt_ret_quote" in detail_df.columns:
        sub = detail_df[detail_df["stage"].isin(
            ["opt_quote_sync_fail", "tpsl_fail", "filled"]
        )]
        if len(sub):
            t_pos = sub["opt_ret_trades"].apply(lambda x: x is not None and float(x) > 0)
            q_pos = sub["opt_ret_quote"].apply(
                lambda x: x is not None and isinstance(x, (int, float)) and float(x) > 0
            )
            sync_cmp = {
                "n_after_stock_sync": int(len(sub)),
                "trades_pos": int(t_pos.sum()),
                "quote_pos": int(q_pos.sum()),
                "trades_only": int((t_pos & ~q_pos).sum()),
                "quote_only": int((q_pos & ~t_pos).sum()),
                "both": int((t_pos & q_pos).sum()),
                "neither": int((~t_pos & ~q_pos).sum()),
            }

    # book quality among entry_probe_ok+
    book_stats: dict[str, Any] = {}
    if not detail_df.empty and "spread_pct" in detail_df.columns:
        okish = detail_df[detail_df["spread_pct"].notna()]
        if len(okish):
            book_stats = {
                "n": int(len(okish)),
                "spread_p50": float(okish["spread_pct"].median()),
                "spread_p90": float(okish["spread_pct"].quantile(0.9)),
                "lag_p50": float(okish["lag_sec"].median()) if "lag_sec" in okish else None,
                "lag_p90": float(okish["lag_sec"].quantile(0.9)) if "lag_sec" in okish else None,
                "mid_p50": float(okish["mid"].median()) if "mid" in okish else None,
            }

    all_c = funnel["ALL"]
    stock_dn = int(all_c.get("stock_dn", 0))
    probe_ok = int(all_c.get("entry_probe_ok", 0))
    filled = int(all_c.get("tpsl_ok", 0))
    trades_only = int(all_c.get("opt_trades_only", 0))
    verdict = {
        "primary_bottleneck": None,
        "notes": [],
    }
    if stock_dn and probe_ok / stock_dn < 0.35:
        verdict["primary_bottleneck"] = "entry_quote_coverage"
        verdict["notes"].append(
            f"entry_probe_ok/stock_dn={probe_ok}/{stock_dn}="
            f"{probe_ok/stock_dn:.1%} — book missing/wide/stale before sync"
        )
    elif trades_only >= max(3, filled):
        verdict["primary_bottleneck"] = "opt_sync_quote_vs_trades"
        verdict["notes"].append(
            f"trades_only={trades_only} vs quote_fill={filled}: "
            "FillSpec lookback sync kills arms that trades mark green"
        )
    elif filled <= 2:
        verdict["primary_bottleneck"] = "opt_sync_or_thin"
        verdict["notes"].append(f"only {filled} quote fills after full sync")
    else:
        verdict["primary_bottleneck"] = "mixed"
        verdict["notes"].append("see funnel_summary.csv")

    summary = {
        "session": "CORE_1030_1130",
        "dir": "DN",
        "morph": "sync",
        "thr": float(args.thr),
        "sync_stock_sec": int(args.sync_stock_sec),
        "sync_opt_sec": int(args.sync_opt_sec),
        "gates": {
            "max_spread_pct": float(args.max_spread_pct),
            "max_lag_sec": float(args.max_lag_sec),
            "min_mid": float(args.min_mid),
            "entry_frac": float(args.entry_frac),
            "exit_frac": float(args.exit_frac),
            "slip_trades": float(args.slip),
        },
        "windows": [list(w) for w in WINDOWS],
        "funnel": {k: dict(v) for k, v in funnel.items()},
        "probe_fail_counts": dict(probe_fail),
        "sync_quote_vs_trades": sync_cmp,
        "book_stats_after_probe": book_stats,
        "verdict": verdict,
        "decision": f"COVERAGE_{str(verdict['primary_bottleneck'] or 'UNKNOWN').upper()}",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print("\n=== FUNNEL ===", flush=True)
    show = [
        c
        for c in summary_df.columns
        if c
        in {
            "window",
            "stock_dn",
            "contract_ok",
            "quote_ticker_ok",
            "entry_probe_ok",
            "stock_sync_ok",
            "opt_quote_sync_ok",
            "tpsl_ok",
            "quote_day_miss",
            "quote_ticker_miss",
            "entry_probe_fail",
            "opt_quote_sync_fail",
            "opt_trades_only",
            "opt_both_pos",
        }
    ]
    print(summary_df[show].to_string(index=False), flush=True)
    print("\nprobe_fail:", dict(probe_fail), flush=True)
    print("sync_quote_vs_trades:", sync_cmp, flush=True)
    print("book:", book_stats, flush=True)
    print("verdict:", summary["decision"], verdict, flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
