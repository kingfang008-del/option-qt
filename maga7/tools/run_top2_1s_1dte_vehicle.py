#!/usr/bin/env python3
"""Top2 + 1s exit clocks × all-1DTE ATM (no fixed T30).

Reuses stock entry/exit from ``top2_1s_parity`` trades_1s.parquet.
Prices ATM call (UP) / put (DN) at open_1dte locks with ask/bid (+ fill075).
Also reports 0DTE on the same clocks as a reference (not the promotion target).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.fills import FillSpec
from maga7.tools.run_smooth_impulse_stock_replay import _equity
from maga7.tools.run_top2_1s_dte_vehicle import _load_lock, _option_ret_from_day, _prep_quotes

NY = "America/New_York"
WINDOWS = [
    {"name": "full_2026", "start": "2026-01-02", "end": "2026-07-17"},
    {"name": "weak_jan_mar", "start": "2026-01-02", "end": "2026-03-31"},
    {"name": "strong_apr_jul", "start": "2026-04-01", "end": "2026-07-17"},
    {"name": "strong_may_jul", "start": "2026-05-01", "end": "2026-07-17"},
]
WD_NAME = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}


def _summarize_opt(df: pd.DataFrame) -> dict:
    ok = df[df["opt_ok"] == True].copy()  # noqa: E712
    miss = int((df["opt_ok"] != True).sum())  # noqa: E712
    if ok.empty:
        return {
            "n": 0,
            "n_missing": miss,
            "fill_rate": 0.0,
            "total_ret": 0.0,
            "maxdd": 0.0,
            "win": None,
            "avg": None,
            "median_hold": None,
        }
    x = ok.copy()
    x["ret"] = pd.to_numeric(x["opt_ret"], errors="coerce")
    eq = _equity(x, frac=0.5)
    return {
        "n": int(len(x)),
        "n_missing": miss,
        "fill_rate": float(len(x) / max(len(df), 1)),
        "total_ret": eq["total_ret"],
        "maxdd": eq["maxdd"],
        "win": eq["trade_win"],
        "avg": float(x["ret"].mean()),
        "median_hold": float(pd.to_numeric(x["hold_minutes"], errors="coerce").median()),
    }


def _summarize_stock(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0, "total_ret": 0.0, "maxdd": 0.0, "win": None, "avg": None, "median_hold": None}
    x = df.copy()
    x["ret"] = pd.to_numeric(x["ret"], errors="coerce")
    eq = _equity(x, frac=0.5)
    return {
        "n": int(len(x)),
        "total_ret": eq["total_ret"],
        "maxdd": eq["maxdd"],
        "win": eq["trade_win"],
        "avg": eq["avg_trade_ret"],
        "median_hold": float(pd.to_numeric(x["hold_minutes"], errors="coerce").median()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default=(
            "maga7/CONFIG/strategy_profiles/"
            "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
        ),
    )
    ap.add_argument(
        "--trades-1s",
        default="/mnt/s990/data/maga7/results/top2_1s_parity_v1/trades_1s.parquet",
    )
    ap.add_argument("--start-date", default="2026-01-02")
    ap.add_argument("--end-date", default="2026-07-17")
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/top2_1s_1dte_vehicle_v1",
    )
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    quote_root = Path(prof["_paths"]["quote_1s_root"])
    lock_path = Path(prof["_paths"]["open_locked_map"]).expanduser()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    trades = pd.read_parquet(args.trades_1s)
    trades["date"] = trades["date"].astype(str)
    trades = trades[
        (trades["date"] >= args.start_date) & (trades["date"] <= args.end_date)
    ].copy()
    trades["weekday"] = pd.to_datetime(trades["date"]).dt.dayofweek
    trades["weekday_name"] = trades["weekday"].map(WD_NAME)

    hold = pd.to_numeric(trades["hold_minutes"], errors="coerce")
    print(
        f"[clocks] n={len(trades)} median_hold={hold.median():.1f}m "
        f"exits={trades['exit_reason'].value_counts().to_dict()}",
        flush=True,
    )

    lock = _load_lock(lock_path)
    fills = {"askbid": FillSpec(1.0, 1.0), "fill075": FillSpec(0.75, 0.75)}
    # Primary: 1DTE; reference: 0DTE on same clocks
    vehicles = [(1, "opt_1dte"), (0, "opt_0dte")]

    rows: list[dict] = []
    qcache: dict[tuple[str, str], pd.DataFrame | None] = {}
    for i, r in enumerate(trades.itertuples(index=False)):
        if i % 50 == 0:
            print(f"[opt] {i}/{len(trades)}", flush=True)
        date, sym = str(r.date), str(r.symbol).upper()
        d = str(r.direction).upper()
        cp = "c" if d == "UP" else "p"
        qkey = (sym, date)
        if qkey not in qcache:
            qp = quote_root / sym / f"{sym}_{date}.parquet"
            if qp.exists():
                qcache[qkey] = _prep_quotes(
                    pd.read_parquet(qp, columns=["timestamp", "ticker", "bid", "ask"])
                )
            else:
                qcache[qkey] = None
        base = {
            "date": date,
            "symbol": sym,
            "direction": d,
            "sleeve": r.sleeve,
            "weekday": int(r.weekday),
            "weekday_name": WD_NAME.get(int(r.weekday), str(r.weekday)),
            "detect_ts": str(r.detect_ts),
            "entry_ts": str(r.entry_ts),
            "exit_ts": str(r.exit_ts),
            "hold_minutes": float(r.hold_minutes),
            "exit_reason": r.exit_reason,
            "stock_ret": float(r.ret),
            "fd_fired": bool(r.fd_fired),
        }
        for dte, veh in vehicles:
            contract = lock.get((sym, date, dte, cp))
            if contract is None:
                for fill_name in fills:
                    rows.append(
                        {
                            **base,
                            "vehicle": veh,
                            "dte": dte,
                            "fill": fill_name,
                            "opt_ok": False,
                            "opt_ret": None,
                            "reason": "no_lock",
                        }
                    )
                continue
            for fill_name, fill in fills.items():
                ores = _option_ret_from_day(
                    qcache[qkey],
                    contract=contract,
                    entry_ts=r.entry_ts,
                    exit_ts=r.exit_ts,
                    fill=fill,
                )
                if ores is None:
                    rows.append(
                        {
                            **base,
                            "vehicle": veh,
                            "dte": dte,
                            "fill": fill_name,
                            "opt_ok": False,
                            "opt_ret": None,
                            "contract": contract,
                            "reason": "no_quote",
                        }
                    )
                else:
                    rows.append(
                        {
                            **base,
                            "vehicle": veh,
                            "dte": dte,
                            "fill": fill_name,
                            "opt_ok": True,
                            "opt_ret": ores["ret"],
                            "contract": ores["contract"],
                            "entry_spread": ores["entry_spread"],
                            "reason": "ok",
                        }
                    )

    odf = pd.DataFrame(rows)
    odf.to_parquet(out / "option_fills.parquet", index=False)
    trades.to_parquet(out / "stock_clocks.parquet", index=False)

    board = []
    for w in WINDOWS:
        st = trades[(trades["date"] >= w["start"]) & (trades["date"] <= w["end"])]
        board.append({"window": w["name"], "vehicle": "stock", "fill": "n/a", **_summarize_stock(st)})
        for veh in ("opt_1dte", "opt_0dte"):
            for fill_name in fills:
                sub = odf[
                    (odf["vehicle"] == veh)
                    & (odf["fill"] == fill_name)
                    & (odf["date"] >= w["start"])
                    & (odf["date"] <= w["end"])
                ]
                board.append({"window": w["name"], "vehicle": veh, "fill": fill_name, **_summarize_opt(sub)})

    # Weekday slice for 1DTE askbid (full 2026)
    wd_rows = []
    for wd, name in WD_NAME.items():
        st = trades[trades["weekday"] == wd]
        wd_rows.append({"weekday": name, "vehicle": "stock", **_summarize_stock(st)})
        for veh in ("opt_1dte", "opt_0dte"):
            sub = odf[(odf["vehicle"] == veh) & (odf["fill"] == "askbid") & (odf["weekday"] == wd)]
            wd_rows.append({"weekday": name, "vehicle": veh, **_summarize_opt(sub)})

    # UP-only 1DTE
    up_board = []
    for w in WINDOWS:
        for fill_name in fills:
            sub = odf[
                (odf["vehicle"] == "opt_1dte")
                & (odf["fill"] == fill_name)
                & (odf["direction"] == "UP")
                & (odf["date"] >= w["start"])
                & (odf["date"] <= w["end"])
            ]
            up_board.append({"window": w["name"], "fill": fill_name, "side": "UP", **_summarize_opt(sub)})

    bdf = pd.DataFrame(board)
    wdf = pd.DataFrame(wd_rows)
    udf = pd.DataFrame(up_board)
    bdf.to_csv(out / "scoreboard.csv", index=False)
    wdf.to_csv(out / "by_weekday.csv", index=False)
    udf.to_csv(out / "up_only_1dte.csv", index=False)

    def cell(window, vehicle, fill="askbid"):
        hit = bdf[(bdf.window == window) & (bdf.vehicle == vehicle) & (bdf.fill == fill)]
        return hit.iloc[0].to_dict() if len(hit) else None

    c1 = cell("strong_may_jul", "opt_1dte")
    c0 = cell("strong_may_jul", "opt_0dte")
    w1 = cell("weak_jan_mar", "opt_1dte")
    s1 = cell("full_2026", "opt_1dte")
    stock_f = cell("full_2026", "stock", "n/a")

    # Verdict: 1DTE useful if may_jul askbid >0 or clearly dominates 0DTE without blowup
    useful = bool(
        c1
        and c1.get("n", 0) >= 15
        and (c1.get("fill_rate") or 0) >= 0.35
        and (
            (c1.get("total_ret") or -1) > 0
            or (
                c0
                and (c1.get("total_ret") or -1) > (c0.get("total_ret") or -1) + 0.1
                and (c1.get("maxdd") or -1) > (c0.get("maxdd") or -1)
            )
        )
    )
    verdict = (
        "1DTE_PROMISING"
        if useful and (c1.get("total_ret") or -1) > 0
        else ("1DTE_LESS_BAD" if useful else "1DTE_NOT_USEFUL")
    )

    summary = {
        "design": "top2_1s_exit_clock__all_1dte_atm__no_fixed_T30",
        "n_clocks": int(len(trades)),
        "verdict": verdict,
        "full_stock": stock_f,
        "full_1dte_askbid": s1,
        "may_jul_1dte_askbid": c1,
        "may_jul_0dte_askbid_ref": c0,
        "jan_mar_1dte_askbid": w1,
        "note": (
            "Primary vehicle=1DTE ATM. 0DTE is reference only. "
            "Exit=1s trail/FD/TIME≤180/EOD (no profile hold_minutes=30). "
            "No new downloads — uses existing open_1dte locks + quote_1s."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    ask = bdf[bdf.fill.isin(["n/a", "askbid"])]
    lines = [
        "# Top2 + 1s Exit × all-1DTE",
        "",
        f"**Verdict: `{verdict}`**",
        "",
        f"- Clocks: `{len(trades)}` · median hold ~{hold.median():.0f}m · **no fixed T30**",
        f"- Primary vehicle: **1DTE ATM** (existing locks/quotes; no re-download)",
        f"- 0DTE shown only as reference",
        "",
        "## Ask/bid scoreboard",
        "",
        "```",
        ask.to_string(index=False),
        "```",
        "",
        "## By weekday (full 2026, ask/bid)",
        "",
        "```",
        wdf.to_string(index=False),
        "```",
        "",
        "## 1DTE UP-only",
        "",
        "```",
        udf.to_string(index=False),
        "```",
        "",
        "## Notes",
        "",
        "- Lock calendar: 1DTE denser on Tue/Thu; 0DTE denser on Mon/Wed/Fri — matches prior weekday effect.",
        "- Same 1s stock exit timestamps for all vehicles.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(ask.to_string(index=False), flush=True)
    print("--- weekday 1dte ---", flush=True)
    print(
        wdf[wdf.vehicle == "opt_1dte"][["weekday", "n", "total_ret", "maxdd", "win", "fill_rate"]].to_string(
            index=False
        ),
        flush=True,
    )
    print("verdict", verdict, flush=True)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
