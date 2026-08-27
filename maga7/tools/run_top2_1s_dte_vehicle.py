#!/usr/bin/env python3
"""Top2 + 1s exit clocks × option vehicle {0DTE, 2DTE} (no fixed T30).

Uses stock entry/exit timestamps from ``top2_1s_parity`` (trail / FD / TIME≤180m / EOD).
Prices ATM call (UP) / ATM put (DN) with ask-in / bid-out. Lock map covers 2026 only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.fills import FillSpec
from maga7.tools.run_smooth_impulse_stock_replay import _equity

NY = "America/New_York"

WINDOWS = [
    {"name": "full_2026", "start": "2026-01-02", "end": "2026-07-17"},
    {"name": "weak_jan_mar", "start": "2026-01-02", "end": "2026-03-31"},
    {"name": "strong_apr_jul", "start": "2026-04-01", "end": "2026-07-17"},
    {"name": "strong_may_jul", "start": "2026-05-01", "end": "2026-07-17"},
]


def _load_lock(path: Path) -> dict[tuple[str, str, int, str], str]:
    """(symbol, date, front_dte, cp) -> contract without O:  cp in {c,p}."""
    df = pd.read_parquet(path)
    out: dict[tuple[str, str, int, str], str] = {}
    for r in df.itertuples(index=False):
        tag = str(getattr(r, "tag", "") or "")
        # prefer explicit ATM tags
        if "_ATM_c" in tag:
            cp = "c"
        elif "_ATM_p" in tag:
            cp = "p"
        else:
            continue
        c = str(r.contract_symbol).replace("O:", "")
        out[(str(r.symbol).upper(), str(r.date_str), int(r.front_dte), cp)] = c
    return out


def _prep_quotes(q: pd.DataFrame) -> pd.DataFrame:
    q = q.copy()
    q["timestamp"] = pd.to_datetime(q["timestamp"])
    if getattr(q["timestamp"].dt, "tz", None) is None:
        q["timestamp"] = q["timestamp"].dt.tz_localize(NY)
    else:
        q["timestamp"] = q["timestamp"].dt.tz_convert(NY)
    q["ticker"] = q["ticker"].astype(str).str.replace("O:", "", regex=False)
    return q.sort_values(["ticker", "timestamp"])


def _option_ret_from_day(
    qday: pd.DataFrame | None,
    *,
    contract: str,
    entry_ts,
    exit_ts,
    fill: FillSpec,
) -> dict | None:
    if qday is None or qday.empty:
        return None
    tkr = str(contract).replace("O:", "")
    sub = qday[qday["ticker"] == tkr]
    if sub.empty:
        return None
    et = pd.Timestamp(entry_ts)
    xt = pd.Timestamp(exit_ts)
    if et.tzinfo is None:
        et = et.tz_localize(NY)
    else:
        et = et.tz_convert(NY)
    if xt.tzinfo is None:
        xt = xt.tz_localize(NY)
    else:
        xt = xt.tz_convert(NY)
    en = sub[sub.timestamp >= et]
    if en.empty:
        return None
    bid0, ask0 = float(en.iloc[0].bid), float(en.iloc[0].ask)
    if not np.isfinite(bid0) or not np.isfinite(ask0) or ask0 <= 0:
        return None
    entry = fill.buy(bid0, ask0)
    ex = sub[sub.timestamp <= xt]
    if ex.empty:
        ex = en
    bid1, ask1 = float(ex.iloc[-1].bid), float(ex.iloc[-1].ask)
    if not np.isfinite(bid1) or not np.isfinite(ask1):
        return None
    exit_px = fill.sell(bid1, ask1)
    if entry <= 0:
        return None
    return {
        "contract": tkr,
        "entry_opt": entry,
        "exit_opt": exit_px,
        "ret": exit_px / entry - 1.0,
        "entry_spread": (ask0 - bid0) / ((ask0 + bid0) / 2) if (ask0 + bid0) > 0 else None,
    }


def _summarize_stock(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0, "total_ret": 0.0, "maxdd": 0.0, "win": None, "avg": None}
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
        help="Stock clock trades from 1s parity (entry/exit_ts)",
    )
    ap.add_argument("--start-date", default="2026-01-02")
    ap.add_argument("--end-date", default="2026-07-17")
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/top2_1s_dte_vehicle_v1",
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
    if trades.empty:
        raise SystemExit("no 1s trades in range")

    # Sanity: no T30 fixed hold in this book
    hold = pd.to_numeric(trades["hold_minutes"], errors="coerce")
    print(
        f"[clocks] n={len(trades)} median_hold={hold.median():.1f}m "
        f"exit={trades['exit_reason'].value_counts().to_dict()}",
        flush=True,
    )

    lock = _load_lock(lock_path)
    fills = {
        "askbid": FillSpec(1.0, 1.0),
        "fill075": FillSpec(0.75, 0.75),
    }

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
                raw = pd.read_parquet(qp, columns=["timestamp", "ticker", "bid", "ask"])
                qcache[qkey] = _prep_quotes(raw)
            else:
                qcache[qkey] = None
        qday = qcache[qkey]
        base = {
            "date": date,
            "symbol": sym,
            "direction": d,
            "sleeve": r.sleeve,
            "detect_ts": str(r.detect_ts),
            "entry_ts": str(r.entry_ts),
            "exit_ts": str(r.exit_ts),
            "hold_minutes": float(r.hold_minutes),
            "exit_reason": r.exit_reason,
            "stock_ret": float(r.ret),
            "fd_fired": bool(r.fd_fired),
        }
        for dte, veh in [(0, "opt_0dte"), (2, "opt_2dte")]:
            contract = lock.get((sym, date, dte, cp))
            if contract is None:
                for fill_name in fills:
                    rows.append(
                        {
                            **base,
                            "vehicle": veh,
                            "fill": fill_name,
                            "opt_ok": False,
                            "opt_ret": None,
                            "reason": "no_lock",
                        }
                    )
                continue
            for fill_name, fill in fills.items():
                ores = _option_ret_from_day(
                    qday,
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
        stock_sm = _summarize_stock(st)
        board.append({"window": w["name"], "vehicle": "stock", "fill": "n/a", **stock_sm})
        for veh in ("opt_0dte", "opt_2dte"):
            for fill_name in fills:
                sub = odf[
                    (odf["vehicle"] == veh)
                    & (odf["fill"] == fill_name)
                    & (odf["date"] >= w["start"])
                    & (odf["date"] <= w["end"])
                ]
                board.append({"window": w["name"], "vehicle": veh, "fill": fill_name, **_summarize_opt(sub)})

    bdf = pd.DataFrame(board)
    bdf.to_csv(out / "scoreboard.csv", index=False)

    # Verdict on askbid May–Jul + Jan–Mar consistency
    def _cell(window, vehicle, fill="askbid"):
        hit = bdf[
            (bdf["window"] == window) & (bdf["vehicle"] == vehicle) & (bdf["fill"] == fill)
        ]
        return hit.iloc[0].to_dict() if len(hit) else None

    c0 = _cell("strong_may_jul", "opt_0dte")
    c2 = _cell("strong_may_jul", "opt_2dte")
    w0 = _cell("weak_jan_mar", "opt_0dte")
    w2 = _cell("weak_jan_mar", "opt_2dte")

    prefer_2dte = False
    if c2 and c0 and w2 and w0:
        # Prefer 2DTE if strong ret not much worse AND weak/maxdd better
        prefer_2dte = bool(
            (c2["total_ret"] >= 0.7 * c0["total_ret"] or c2["maxdd"] > c0["maxdd"] + 0.05)
            and (w2["maxdd"] >= w0["maxdd"] - 0.02)
            and (c2.get("fill_rate") or 0) >= 0.5
            and (w2.get("fill_rate") or 0) >= 0.5
        )

    summary = {
        "design": "top2_detect_1m__exit_1s_clock__atm_option_askbid__no_fixed_T30",
        "trades_1s": str(args.trades_1s),
        "n_clocks": int(len(trades)),
        "lock_range": "2026-01-02..2026-07-17",
        "prefer_2dte_vs_0dte_askbid": prefer_2dte,
        "scoreboard": board,
        "may_jul_askbid": {"0dte": c0, "2dte": c2},
        "jan_mar_askbid": {"0dte": w0, "2dte": w2},
        "note": (
            "Exit clock from stock 1s trail/FD/TIME(180m)/EOD — not profile hold_minutes=30. "
            "ATM call UP / ATM put DN from open lock. Missing locks/quotes counted in n_missing."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    show = bdf[
        (bdf["fill"].isin(["askbid", "n/a"]))
        | ((bdf["vehicle"] != "stock") & (bdf["fill"] == "fill075"))
    ]
    # keep askbid + stock + fill075 for options
    lines = [
        "# Top2 + 1s Exit × 0DTE / 2DTE",
        "",
        f"**Clocks:** `{len(trades)}` Top2 seats with 1s stock exits (no fixed T30).",
        f"**Prefer 2DTE (heuristic): `{prefer_2dte}`**",
        "",
        "## Scoreboard",
        "",
        "```",
        bdf.to_string(index=False),
        "```",
        "",
        "## May–Jul ask/bid",
        "",
        f"- 0DTE: `{c0}`",
        f"- 2DTE: `{c2}`",
        "",
        "## Jan–Mar ask/bid",
        "",
        f"- 0DTE: `{w0}`",
        f"- 2DTE: `{w2}`",
        "",
        "## Notes",
        "",
        "- Same entry/exit timestamps for stock and both DTE vehicles.",
        "- Ask/bid = opponent fill; fill075 = research median fill.",
        "- Lock map only 2026 YTD — not 2024–2025.",
        "- TIME exit at 180m is a no-development cap, not the old T30 hold.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(bdf[bdf["fill"].isin(["n/a", "askbid"])].to_string(index=False), flush=True)
    print("prefer_2dte", prefer_2dte, flush=True)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
