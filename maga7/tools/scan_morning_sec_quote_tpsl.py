#!/usr/bin/env python3
"""Morning sec-MF edges + quote FillSpec TP/SL (executable gates, dual-window).

Reads ``research_morn_sec_edge_*`` events (causal 1s stock). Entry gates:
quote lag / spread / mid. Exit: first-passage TP/SL on sell-mark.
``max_hold_sec`` is safety only — not the alpha clock.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_morning_sec_quote_tpsl \\
    --events-tags research_morn_sec_edge_may_jul,research_morn_sec_edge_jan_mar \\
    --tag research_morn_sec_quote_tpsl_dual
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
from maga7.tools.scan_launch_slope_tpsl import _port

FREEZE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)

# Entry cells (no clock H). Includes prior sec-option-fill candidates + nearby.
DEFAULT_CELLS: list[dict[str, Any]] = [
    {
        "name": "w100_s20_fp005_vz1_p0",
        "mf_window_sec": 100,
        "streak_min": 20,
        "from_prev_min": 0.005,
        "vol_z_min": 1.0,
        "peer_min": 0,
    },
    {
        "name": "w100_s20_fp005_vz1_p3",
        "mf_window_sec": 100,
        "streak_min": 20,
        "from_prev_min": 0.005,
        "vol_z_min": 1.0,
        "peer_min": 3,
    },
    {
        "name": "w100_s20_fp003_vz1_p2",
        "mf_window_sec": 100,
        "streak_min": 20,
        "from_prev_min": 0.003,
        "vol_z_min": 1.0,
        "peer_min": 2,
    },
    {
        "name": "w100_s40_fp005_vz1_p2",
        "mf_window_sec": 100,
        "streak_min": 40,
        "from_prev_min": 0.005,
        "vol_z_min": 1.0,
        "peer_min": 2,
    },
    {
        "name": "w60_s20_fp005_vz1_p2",
        "mf_window_sec": 60,
        "streak_min": 20,
        "from_prev_min": 0.005,
        "vol_z_min": 1.0,
        "peer_min": 2,
    },
    {
        "name": "w60_s40_fp003_vz0_p3",
        "mf_window_sec": 60,
        "streak_min": 40,
        "from_prev_min": 0.003,
        "vol_z_min": 0.0,
        "peer_min": 3,
    },
    {
        "name": "w180_s40_fp005_vz1_p2",
        "mf_window_sec": 180,
        "streak_min": 40,
        "from_prev_min": 0.005,
        "vol_z_min": 1.0,
        "peer_min": 2,
    },
    {
        "name": "w180_s60_fp005_vz1_p3",
        "mf_window_sec": 180,
        "streak_min": 60,
        "from_prev_min": 0.005,
        "vol_z_min": 1.0,
        "peer_min": 3,
    },
]


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
        ["date", "symbol", "dir", "ts", "mf_window_sec", "streak_min"]
    ).reset_index(drop=True)


def _filter_events(events: pd.DataFrame, cell: dict[str, Any]) -> pd.DataFrame:
    w = int(cell["mf_window_sec"])
    s = int(cell["streak_min"])
    fp = float(cell["from_prev_min"])
    vz = float(cell["vol_z_min"])
    peer = int(cell["peer_min"])
    sub = events[
        (events["mf_window_sec"] == w) & (events["streak_min"] == s)
    ].copy()
    if sub.empty:
        return sub
    up_ok = (sub["dir"] == "UP") & (sub["from_prev"] >= fp)
    dn_ok = (sub["dir"] == "DN") & (sub["from_prev"] <= -fp)
    sub = sub[up_ok | dn_ok]
    sub = sub[(sub["vol_z"].isna()) | (sub["vol_z"] >= vz)]
    sub = sub[sub["peer_n"] >= peer]
    return sub.sort_values("ts").drop_duplicates(
        ["date", "symbol", "dir", "mf_window_sec", "streak_min"], keep="first"
    ).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=FREEZE)
    ap.add_argument(
        "--events-tags",
        default="research_morn_sec_edge_may_jul,research_morn_sec_edge_jan_mar",
    )
    ap.add_argument("--tag", default="research_morn_sec_quote_tpsl_dual")
    ap.add_argument("--cells", default="")
    ap.add_argument("--tps", default="0.10,0.15,0.20")
    ap.add_argument("--sls", default="0.10,0.15,0.25")
    ap.add_argument("--max-spreads", default="0.08,0.10,0.15")
    ap.add_argument("--max-lags", default="2,3")
    ap.add_argument("--min-mid", type=float, default=0.05)
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--entry-frac", type=float, default=0.75)
    ap.add_argument("--exit-frac", type=float, default=0.75)
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    results_dir = Path(paths["results_dir"])
    quote_root = Path(paths["quote_1s_root"])
    fill = FillSpec(entry_frac=float(args.entry_frac), exit_frac=float(args.exit_frac))

    cells = list(DEFAULT_CELLS)
    if args.cells.strip():
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
            ticker, dte, _ = resolve_open_lock_contract(
                multi_idx.get((sym, date)),
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
                                if float(f["entry_spread_pct"]) > max_sp:
                                    continue
                                if float(f["entry_lag_sec"]) > max_lag:
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
                            score_rows.append(
                                {
                                    "events_tag": etag,
                                    "window": window,
                                    "cell": cell["name"],
                                    "mf_window_sec": cell["mf_window_sec"],
                                    "streak_min": cell["streak_min"],
                                    "from_prev_min": cell["from_prev_min"],
                                    "vol_z_min": cell["vol_z_min"],
                                    "peer_min": cell["peer_min"],
                                    "max_spread_pct": max_sp,
                                    "max_lag_sec": max_lag,
                                    "min_mid": float(args.min_mid),
                                    "n_signals": int(len(sigs)),
                                    "tp": tp,
                                    "sl": sl,
                                    "max_hold_sec": int(args.max_hold_sec),
                                    **st,
                                }
                            )
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
        ok = score[
            (score["mean"].fillna(-1) > 0)
            & (score["add"].fillna(0) > 0)
            & (score["day_win"].fillna(0) >= 0.55)
            & (score["n"].fillna(0) >= 20)
            & (score["frac_max_hold"].fillna(1) <= 0.50)
        ].sort_values(["events_tag", "add"], ascending=[True, False])
        picks = ok.to_dict(orient="records")

        if len(event_tags) >= 2:
            keys = ["cell", "max_spread_pct", "max_lag_sec", "tp", "sl"]
            cols = keys + ["n", "mean", "add", "day_win", "frac_max_hold"]
            a = score[score["events_tag"] == event_tags[0]][cols].copy()
            b = score[score["events_tag"] == event_tags[1]][cols].copy()
            merged = a.merge(b, on=keys, suffixes=("_w0", "_w1"))
            for _, r in merged.iterrows():
                if (
                    float(r["n_w0"]) >= 20
                    and float(r["n_w1"]) >= 20
                    and float(r["mean_w0"]) > 0
                    and float(r["mean_w1"]) > 0
                    and float(r["add_w0"]) > 0
                    and float(r["add_w1"]) > 0
                    and float(r["day_win_w0"]) >= 0.55
                    and float(r["day_win_w1"]) >= 0.55
                    and float(r["frac_max_hold_w0"]) <= 0.50
                    and float(r["frac_max_hold_w1"]) <= 0.50
                ):
                    dual_ok.append(
                        {
                            "cell": r["cell"],
                            "max_spread_pct": float(r["max_spread_pct"]),
                            "max_lag_sec": float(r["max_lag_sec"]),
                            "tp": float(r["tp"]),
                            "sl": float(r["sl"]),
                            "w0": event_tags[0],
                            "w0_n": int(r["n_w0"]),
                            "w0_mean": float(r["mean_w0"]),
                            "w0_add": float(r["add_w0"]),
                            "w0_day_win": float(r["day_win_w0"]),
                            "w1": event_tags[1],
                            "w1_n": int(r["n_w1"]),
                            "w1_mean": float(r["mean_w1"]),
                            "w1_add": float(r["add_w1"]),
                            "w1_day_win": float(r["day_win_w1"]),
                            "add_sum": float(r["add_w0"]) + float(r["add_w1"]),
                        }
                    )
            dual_ok.sort(key=lambda x: x["add_sum"], reverse=True)

    summary = {
        "events_tags": event_tags,
        "book": "quote_fill_tpsl",
        "entry": "morning_sec_mf_streak",
        "entry_gates": {
            "max_spreads": spreads,
            "max_lags": lags,
            "min_mid": float(args.min_mid),
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
        "verdict": "PASS" if dual_ok else "REJECT",
        "note": (
            "Morning sec-MF streak on causal 1s; quote FillSpec TP/SL dual-window. "
            "Independent of launch_slope / MOM60."
        ),
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
            "add",
            "day_win",
            "frac_tp",
            "frac_sl",
        ]
        print("\n=== top by add ===", flush=True)
        print(score.sort_values("add", ascending=False)[cols].head(20).to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
