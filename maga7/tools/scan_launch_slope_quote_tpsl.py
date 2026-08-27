#!/usr/bin/env python3
"""Launch-slope + quote FillSpec TP/SL with executable entry gates.

Entry gates (causal): quote within ``max_lag_sec``, bid/ask valid,
``spread_pct <= max_spread``, ``mid >= min_mid``. Exit = first-passage TP/SL
on sell-mark; ``max_hold_sec`` safety only.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_launch_slope_quote_tpsl \\
    --events-tags research_launch_slope_may_jul,research_launch_slope_jan_mar_am \\
    --tag research_launch_slope_quote_tpsl_dual
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
from maga7.common.open_lock import (
    load_multidte_lock_index,
    resolve_open_lock_contract,
    resolve_otm_rungs,
)
from maga7.common.option_quote_tpsl import entry_quote_row, simulate_quote_tpsl
from maga7.common.replay import load_quotes, path_for_ticker, to_ny
from maga7.tools.scan_launch_slope_tpsl import DEFAULT_CELLS, _filter_events, _port

FREEZE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)

OPEN_CELLS = [c["name"] for c in DEFAULT_CELLS if str(c["session"]).startswith("open")]


def _load_events(results_dir: Path, tag: str) -> pd.DataFrame:
    p = results_dir / tag / "events.parquet"
    if p.is_file():
        events = pd.read_parquet(p)
    else:
        csv_path = results_dir / tag / "events.csv"
        if not csv_path.is_file():
            raise SystemExit(f"missing events: {p}")
        events = pd.read_csv(csv_path)
    return events.drop_duplicates(
        ["date", "symbol", "dir", "ts", "session", "slope_sec", "abs_ret_min"]
    ).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=FREEZE)
    ap.add_argument(
        "--events-tags",
        default="research_launch_slope_may_jul,research_launch_slope_jan_mar_am",
    )
    ap.add_argument("--tag", default="research_launch_slope_quote_tpsl_dual")
    ap.add_argument("--cells", default=",".join(OPEN_CELLS))
    ap.add_argument("--tps", default="0.10,0.15,0.20")
    ap.add_argument("--sls", default="0.10,0.15,0.25")
    ap.add_argument("--max-spreads", default="0.05,0.08,0.10,0.15,0.25")
    ap.add_argument("--max-lags", default="2,3,5")
    ap.add_argument("--min-mid", type=float, default=0.10)
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--entry-frac", type=float, default=0.75)
    ap.add_argument("--exit-frac", type=float, default=0.75)
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    results_dir = Path(paths["results_dir"])
    quote_root = Path(paths["quote_1s_root"])
    fill = FillSpec(entry_frac=float(args.entry_frac), exit_frac=float(args.exit_frac))

    want = {x.strip() for x in args.cells.split(",") if x.strip()}
    cells = [c for c in DEFAULT_CELLS if c["name"] in want]
    if not cells:
        raise SystemExit(f"no cells matched {want}")

    tps = [float(x) for x in args.tps.split(",") if x.strip()]
    sls = [float(x) for x in args.sls.split(",") if x.strip()]
    spreads = [float(x) for x in args.max_spreads.split(",") if x.strip()]
    lags = [float(x) for x in args.max_lags.split(",") if x.strip()]
    event_tags = [x.strip() for x in args.events_tags.split(",") if x.strip()]

    lock_path = Path(paths.get("open_locked_map") or paths.get("locked_map")).expanduser()
    multi_idx = load_multidte_lock_index(lock_path) if lock_path.is_file() else {}
    otm_rungs = resolve_otm_rungs(prof, default=3)

    out = results_dir / args.tag
    out.mkdir(parents=True, exist_ok=True)
    score_rows: list[dict[str, Any]] = []
    best_trades: dict[str, pd.DataFrame] = {}

    # Widest gate used while caching resolved quote paths.
    max_spread_cache = max(spreads)
    max_lag_cache = max(lags)

    for etag in event_tags:
        events = _load_events(results_dir, etag)
        dates = sorted(events["date"].astype(str).unique())
        window = f"{dates[0]}_{dates[-1]}" if dates else etag
        print(f"\n=== events={etag} window={window} cells={len(cells)} ===", flush=True)

        union_meta: dict[tuple[str, str, str, str], float | None] = {}
        cell_sigs: dict[str, pd.DataFrame] = {}
        for cell in cells:
            sigs = _filter_events(events, cell)
            cell_sigs[cell["name"]] = sigs
            for _, r in sigs.iterrows():
                k = (str(r["date"]), str(r["symbol"]), str(r["dir"]), str(r["ts"]))
                px = float(r["entry_px"]) if "entry_px" in r and pd.notna(r["entry_px"]) else None
                union_meta[k] = px

        quote_cache: dict[tuple[str, str], pd.DataFrame | None] = {}
        # key -> path df + spot meta (resolved ticker)
        resolved: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        n_miss = 0
        for i, ((date, sym, direction, ts_s), spot) in enumerate(sorted(union_meta.items())):
            if i % 100 == 0:
                print(
                    f"[{etag} resolve] {i}/{len(union_meta)} ok={len(resolved)} miss={n_miss}",
                    flush=True,
                )
            qkey = (sym, date)
            if qkey not in quote_cache:
                quote_cache[qkey] = load_quotes(quote_root, sym, date)
            qday = quote_cache[qkey]
            if qday is None or qday.empty:
                n_miss += 1
                continue
            by_dte = multi_idx.get((sym, date))
            ticker, dte, _ = resolve_open_lock_contract(
                by_dte,
                direction=direction,
                moneyness="ATM",
                spot=spot,
                prefer_dte=0,
                allowed_dte=[0, 1, 2],
                clear_otm_thresh=0.01,
                ladder=True,
                otm_rungs=otm_rungs,
            )
            if not ticker:
                n_miss += 1
                continue
            path = path_for_ticker(qday, ticker)
            if path is None or path.empty:
                n_miss += 1
                continue
            # Pre-check with widest gate so we keep candidates for tighter grids.
            probe = entry_quote_row(
                path,
                to_ny(ts_s),
                max_lag_sec=max_lag_cache,
                max_spread_pct=max_spread_cache,
                min_mid=float(args.min_mid),
            )
            if probe is None:
                n_miss += 1
                continue
            resolved[(date, sym, direction, ts_s)] = {
                "date": date,
                "symbol": sym,
                "dir": direction,
                "sig_ts": to_ny(ts_s),
                "ticker": ticker,
                "dte": dte,
                "path": path,
                "entry_spread_pct": probe["spread_pct"],
                "entry_lag_sec": probe["lag_sec"],
                "entry_mid": probe["mid"],
            }

        print(
            f"[{etag}] unique={len(union_meta)} resolved_wide={len(resolved)} miss={n_miss}",
            flush=True,
        )

        for cell in cells:
            sigs = cell_sigs[cell["name"]]
            for max_sp in spreads:
                for max_lag in lags:
                    for tp in tps:
                        for sl in sls:
                            raw: list[dict[str, Any]] = []
                            n_gate = 0
                            for _, r in sigs.iterrows():
                                k = (
                                    str(r["date"]),
                                    str(r["symbol"]),
                                    str(r["dir"]),
                                    str(r["ts"]),
                                )
                                f = resolved.get(k)
                                if f is None:
                                    continue
                                # Tight gate vs cached wide resolve.
                                if float(f["entry_spread_pct"]) > max_sp:
                                    n_gate += 1
                                    continue
                                if float(f["entry_lag_sec"]) > max_lag:
                                    n_gate += 1
                                    continue
                                sim = simulate_quote_tpsl(
                                    f["path"],
                                    f["sig_ts"],
                                    tp=tp,
                                    sl=sl,
                                    max_hold_sec=int(args.max_hold_sec),
                                    fill=fill,
                                    max_lag_sec=max_lag,
                                    max_spread_pct=max_sp,
                                    min_mid=float(args.min_mid),
                                )
                                if sim is None or not np.isfinite(sim["ret"]):
                                    n_gate += 1
                                    continue
                                raw.append(
                                    {
                                        "date": f["date"],
                                        "symbol": f["symbol"],
                                        "dir": f["dir"],
                                        "entry_ts": str(sim["entry_ts"]),
                                        "exit_ts": str(sim["exit_ts"]),
                                        "ticker": f["ticker"],
                                        "dte": f["dte"],
                                        "ret": sim["ret"],
                                        "exit_reason": sim["reason"],
                                        "hold_sec": sim["hold_sec"],
                                        "entry_spread_pct": sim["entry_spread_pct"],
                                        "entry_lag_sec": sim["entry_lag_sec"],
                                    }
                                )
                            st = _port(raw)
                            row = {
                                "events_tag": etag,
                                "window": window,
                                "cell": cell["name"],
                                "session": cell["session"],
                                "max_spread_pct": max_sp,
                                "max_lag_sec": max_lag,
                                "min_mid": float(args.min_mid),
                                "n_signals": int(len(sigs)),
                                "n_gate_skip": int(n_gate),
                                "tp": tp,
                                "sl": sl,
                                "max_hold_sec": int(args.max_hold_sec),
                                **st,
                            }
                            score_rows.append(row)
                            key = f"{etag}|{cell['name']}|sp{max_sp}|lag{max_lag}|tp{tp}|sl{sl}"
                            if st.get("n", 0) > 0 and st.get("mean") is not None and st["mean"] > 0:
                                best_trades[key] = pd.DataFrame(raw)
                            if st.get("n", 0) >= 15:
                                print(
                                    f"[{etag} {cell['name']} sp≤{max_sp} lag≤{max_lag} "
                                    f"tp{tp}/sl{sl}] n={st['n']} mean={st['mean']} "
                                    f"add={st['add']:+.3f} day_win={st['day_win']}",
                                    flush=True,
                                )

    score = pd.DataFrame(score_rows)
    score.to_csv(out / "scoreboard.csv", index=False)

    picks: list[dict[str, Any]] = []
    dual_ok: list[dict[str, Any]] = []
    if len(score):
        # Per-window pick gate.
        ok = score[
            (score["mean"].fillna(-1) > 0)
            & (score["add"].fillna(0) > 0)
            & (score["day_win"].fillna(0) >= 0.55)
            & (score["n"].fillna(0) >= 20)
            & (score["frac_max_hold"].fillna(1) <= 0.50)
        ].sort_values(["events_tag", "add"], ascending=[True, False])
        picks = ok.to_dict(orient="records")

        # Dual-window: same (cell, sp, lag, tp, sl) positive on BOTH tags.
        if len(event_tags) >= 2:
            keys = ["cell", "max_spread_pct", "max_lag_sec", "tp", "sl"]
            a = score[score["events_tag"] == event_tags[0]].set_index(keys)
            b = score[score["events_tag"] == event_tags[1]].set_index(keys)
            common = a.index.intersection(b.index)
            for idx in common:
                ra, rb = a.loc[idx], b.loc[idx]
                # handle duplicate index if any
                if isinstance(ra, pd.DataFrame):
                    ra = ra.iloc[0]
                if isinstance(rb, pd.DataFrame):
                    rb = rb.iloc[0]
                if (
                    float(ra.get("n") or 0) >= 20
                    and float(rb.get("n") or 0) >= 20
                    and float(ra.get("mean") or -1) > 0
                    and float(rb.get("mean") or -1) > 0
                    and float(ra.get("add") or 0) > 0
                    and float(rb.get("add") or 0) > 0
                    and float(ra.get("day_win") or 0) >= 0.55
                    and float(rb.get("day_win") or 0) >= 0.55
                    and ra.get("frac_max_hold") is not None
                    and rb.get("frac_max_hold") is not None
                    and float(ra["frac_max_hold"]) <= 0.50
                    and float(rb["frac_max_hold"]) <= 0.50
                ):
                    dual_ok.append(
                        {
                            "cell": idx[0],
                            "max_spread_pct": idx[1],
                            "max_lag_sec": idx[2],
                            "tp": idx[3],
                            "sl": idx[4],
                            "w0": event_tags[0],
                            "w0_n": int(ra["n"]),
                            "w0_mean": float(ra["mean"]),
                            "w0_add": float(ra["add"]),
                            "w0_day_win": float(ra["day_win"]),
                            "w1": event_tags[1],
                            "w1_n": int(rb["n"]),
                            "w1_mean": float(rb["mean"]),
                            "w1_add": float(rb["add"]),
                            "w1_day_win": float(rb["day_win"]),
                            "add_sum": float(ra["add"]) + float(rb["add"]),
                        }
                    )
            dual_ok.sort(key=lambda x: x["add_sum"], reverse=True)

        for i, p in enumerate(picks[:8]):
            key = (
                f"{p['events_tag']}|{p['cell']}|sp{p['max_spread_pct']}|"
                f"lag{p['max_lag_sec']}|tp{p['tp']}|sl{p['sl']}"
            )
            if key in best_trades:
                best_trades[key].to_csv(
                    out
                    / (
                        f"trades_pick{i}_{p['events_tag']}_{p['cell']}_"
                        f"sp{p['max_spread_pct']}_tp{p['tp']}_sl{p['sl']}.csv"
                    ),
                    index=False,
                )

    summary = {
        "events_tags": event_tags,
        "book": "quote_fill_tpsl",
        "entry_gates": {
            "max_spreads": spreads,
            "max_lags": lags,
            "min_mid": float(args.min_mid),
            "fill": {"entry_frac": float(args.entry_frac), "exit_frac": float(args.exit_frac)},
        },
        "tps": tps,
        "sls": sls,
        "n_score_rows": int(len(score)),
        "n_picks_any_window": int(len(picks)),
        "n_dual_window_pass": int(len(dual_ok)),
        "dual_window_pass": dual_ok[:20],
        "picks_any_window": picks[:30],
        "top_by_add": (
            score.sort_values("add", ascending=False).head(25).to_dict(orient="records")
            if len(score)
            else []
        ),
        "note": (
            "Executable quote gates + FillSpec TP/SL. Dual pass requires both "
            "event tags to clear mean/add/day_win/n/frac_max_hold gates."
        ),
        "verdict": "PASS" if dual_ok else "REJECT",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "dual_pass.json").write_text(json.dumps(dual_ok[:30], indent=2, default=str), encoding="utf-8")
    (out / "picks.json").write_text(json.dumps(picks[:30], indent=2, default=str), encoding="utf-8")

    print(f"\n=== dual-window PASS ({len(dual_ok)}) verdict={summary['verdict']} ===", flush=True)
    print(json.dumps(dual_ok[:10], indent=2, default=str), flush=True)
    if len(score):
        cols = [
            "events_tag",
            "cell",
            "max_spread_pct",
            "max_lag_sec",
            "tp",
            "sl",
            "n",
            "mean",
            "win",
            "add",
            "day_win",
            "frac_tp",
            "frac_sl",
            "frac_max_hold",
        ]
        print("\n=== top by add ===", flush=True)
        print(score.sort_values("add", ascending=False)[cols].head(20).to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
