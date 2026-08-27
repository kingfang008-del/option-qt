#!/usr/bin/env python3
"""UP smooth+impulse entries: QQQ align × vehicle (stock / 0DTE / 2DTE).

Same detect clock → stock trail120 path defines exit_ts → price ATM call
(0DTE vs 2DTE) ask-in/bid-out at those stamps. Research only.
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
from maga7.common.signals import attach_mf_features, load_stock_month_files
from maga7.common.smooth_trend import (
    ImpulseLaunchConfig,
    SmoothLaunchConfig,
    SmoothStockTradeConfig,
    apply_day_portfolio_cap,
    replay_smooth_impulse_stock_day,
)
from maga7.tools.run_smooth_impulse_stock_replay import SYMS, MONTHS, _equity

NY = "America/New_York"


def _load_lock(path: Path) -> dict[tuple[str, str, int, int], str]:
    """(symbol, date, front_dte, bucket_id) -> contract without O:"""
    df = pd.read_parquet(path)
    out: dict[tuple[str, str, int, int], str] = {}
    for r in df.itertuples():
        c = str(r.contract_symbol).replace("O:", "")
        out[(str(r.symbol).upper(), str(r.date_str), int(r.front_dte), int(r.bucket_id))] = c
    return out


def _qqq_signed(qqq_day: pd.DataFrame, ts: pd.Timestamp, *, mode: str) -> float | None:
    d = qqq_day.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    if d["timestamp"].dt.tz is None:
        d["timestamp"] = d["timestamp"].dt.tz_localize(NY)
    else:
        d["timestamp"] = d["timestamp"].dt.tz_convert(NY)
    ts = pd.Timestamp(ts).tz_convert(NY) if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts).tz_localize(NY)
    upto = d[d.timestamp <= ts]
    if upto.empty:
        return None
    if mode == "from_open":
        c0 = float(d.iloc[0]["close"])
        c1 = float(upto.iloc[-1]["close"])
        return c1 / c0 - 1.0 if c0 > 0 else None
    if mode == "last10":
        w = upto[upto.timestamp >= ts - pd.Timedelta(minutes=10)]
        if len(w) < 2:
            return None
        c0 = float(w.iloc[0]["close"])
        c1 = float(w.iloc[-1]["close"])
        return c1 / c0 - 1.0 if c0 > 0 else None
    return None


def _option_ret(
    quote_root: Path,
    *,
    symbol: str,
    date: str,
    contract: str,
    entry_ts: pd.Timestamp,
    exit_ts: pd.Timestamp,
    fill: FillSpec,
) -> dict | None:
    p = quote_root / symbol / f"{symbol}_{date}.parquet"
    if not p.exists():
        return None
    q = pd.read_parquet(p)
    q["timestamp"] = pd.to_datetime(q["timestamp"])
    if q["timestamp"].dt.tz is None:
        q["timestamp"] = q["timestamp"].dt.tz_localize(NY)
    else:
        q["timestamp"] = q["timestamp"].dt.tz_convert(NY)
    tkr = str(contract).replace("O:", "")
    sub = q[q["ticker"].astype(str).str.replace("O:", "", regex=False) == tkr].sort_values("timestamp")
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default=(
            "maga7/CONFIG/strategy_profiles/"
            "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
        ),
    )
    ap.add_argument("--start-date", default="2026-05-01")
    ap.add_argument("--end-date", default="2026-07-17")
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/research_smooth_qqq_weekly_may_jul",
    )
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    stock_root = Path(prof["_paths"]["stock_root"]).expanduser()
    quote_root = Path(prof["_paths"]["quote_1s_root"])
    lock_path = Path(prof["_paths"]["open_locked_map"]).expanduser()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lock = _load_lock(lock_path)
    fill = FillSpec(1.0, 1.0)  # ask/bid stress

    print("[load] stocks", flush=True)
    data: dict[str, pd.DataFrame] = {}
    for sym in SYMS + ["QQQ"]:
        raw = load_stock_month_files(stock_root, sym, MONTHS)
        if raw.empty:
            continue
        data[sym] = attach_mf_features(raw)
        data[sym] = data[sym][data[sym]["date"].astype(str).between(args.start_date, args.end_date)]

    smooth_cfg = SmoothLaunchConfig(scan_end="11:30", min_look_ret=0.002, cooldown_minutes=60)
    impulse_cfg = ImpulseLaunchConfig(scan_end="11:30", min_look_ret=0.004)
    trade_cfg = SmoothStockTradeConfig(
        break_max_adverse=0.012,
        max_hold_minutes=180,
        break_min_up_frac=0.35,
        first_per_symbol_dir=True,
    )

    # Build UP stock trades first
    base_trades: list[dict] = []
    for sym in SYMS:
        raw = data.get(sym)
        if raw is None or raw.empty:
            continue
        print(f"[entries] {sym}", flush=True)
        for date in sorted(raw["date"].astype(str).unique()):
            day = raw[raw["date"].astype(str) == date]
            rows = replay_smooth_impulse_stock_day(
                day,
                symbol=sym,
                date=date,
                smooth_cfg=smooth_cfg,
                impulse_cfg=impulse_cfg,
                trade_cfg=trade_cfg,
            )
            rows = [r for r in rows if r["direction"] == "UP"]
            # attach QQQ features
            qday = data.get("QQQ")
            qday_d = None
            if qday is not None:
                qday_d = qday[qday["date"].astype(str) == date]
            for r in rows:
                ts = pd.Timestamp(r["detect_ts"])
                r["qqq_from_open"] = (
                    _qqq_signed(qday_d, ts, mode="from_open") if qday_d is not None and len(qday_d) else None
                )
                r["qqq_last10"] = (
                    _qqq_signed(qday_d, ts, mode="last10") if qday_d is not None and len(qday_d) else None
                )
            base_trades.extend(rows)

    capped = apply_day_portfolio_cap(base_trades, max_positions=2)
    bdf = pd.DataFrame(capped)
    bdf.to_csv(out / "entries_up.csv", index=False)

    # Option fills on same clocks
    opt_rows = []
    for r in capped:
        date, sym = str(r["date"]), str(r["symbol"])
        for dte, label in [(0, "opt_0dte"), (2, "opt_2dte")]:
            # ATM call bucket_id=2
            contract = lock.get((sym, date, dte, 2))
            if contract is None:
                opt_rows.append({**r, "vehicle": label, "opt_ret": None, "opt_ok": False, "reason": "no_lock"})
                continue
            ores = _option_ret(
                quote_root,
                symbol=sym,
                date=date,
                contract=contract,
                entry_ts=r["entry_ts"],
                exit_ts=r["exit_ts"],
                fill=fill,
            )
            if ores is None:
                opt_rows.append({**r, "vehicle": label, "opt_ret": None, "opt_ok": False, "reason": "no_quote"})
            else:
                opt_rows.append(
                    {
                        **r,
                        "vehicle": label,
                        "opt_ret": ores["ret"],
                        "opt_ok": True,
                        "contract": ores["contract"],
                        "entry_spread": ores["entry_spread"],
                        "reason": "ok",
                    }
                )
    odf = pd.DataFrame(opt_rows)
    odf.to_csv(out / "option_fills.csv", index=False)

    def stock_equity(df: pd.DataFrame) -> dict:
        if df.empty:
            return {"total_ret": 0.0, "maxdd": 0.0, "n_trades": 0, "trade_win": None}
        x = df.copy()
        x["ret"] = pd.to_numeric(x["ret"], errors="coerce")
        return {k: v for k, v in _equity(x, frac=0.5).items() if k != "daily"}

    def opt_equity(df: pd.DataFrame) -> dict:
        ok = df[df["opt_ok"] == True].copy()  # noqa: E712
        if ok.empty:
            return {"total_ret": 0.0, "maxdd": 0.0, "n_trades": 0, "trade_win": None, "n_missing": int(len(df))}
        x = ok.copy()
        x["ret"] = pd.to_numeric(x["opt_ret"], errors="coerce")
        eq = {k: v for k, v in _equity(x, frac=0.5).items() if k != "daily"}
        eq["n_missing"] = int((df["opt_ok"] != True).sum())  # noqa: E712
        eq["avg_trade_ret"] = float(x["ret"].mean())
        return eq

    variants = []
    filters = [
        ("all", lambda d: d),
        ("qqq_from_open>0", lambda d: d[pd.to_numeric(d["qqq_from_open"], errors="coerce") > 0]),
        ("qqq_last10>0", lambda d: d[pd.to_numeric(d["qqq_last10"], errors="coerce") > 0]),
        (
            "qqq_both>0",
            lambda d: d[
                (pd.to_numeric(d["qqq_from_open"], errors="coerce") > 0)
                & (pd.to_numeric(d["qqq_last10"], errors="coerce") > 0)
            ],
        ),
    ]
    for fname, ffn in filters:
        sdf = ffn(bdf)
        s_eq = stock_equity(sdf)
        variants.append({"filter": fname, "vehicle": "stock", **s_eq})
        for veh in ("opt_0dte", "opt_2dte"):
            osub = odf[(odf["vehicle"] == veh) & (odf["date"].isin(set(sdf["date"])) if len(sdf) else False)]
            # re-filter by entry keys
            if len(sdf):
                keys = set(zip(sdf["date"], sdf["symbol"], sdf["detect_ts"]))
                osub = odf[odf["vehicle"] == veh]
                osub = osub[
                    osub.apply(lambda r: (r["date"], r["symbol"], r["detect_ts"]) in keys, axis=1)
                ]
            else:
                osub = odf.iloc[0:0]
            o_eq = opt_equity(osub)
            variants.append({"filter": fname, "vehicle": veh, **o_eq})

    vdf = pd.DataFrame(variants)
    vdf.to_csv(out / "scoreboard.csv", index=False)

    # best by total_ret among stock / 2dte with enough trades
    cand = vdf[(vdf["n_trades"] >= 30)].copy()
    best = None
    if len(cand):
        best = cand.sort_values("total_ret", ascending=False).iloc[0].to_dict()

    summary = {
        "n_entries_up_capped": int(len(bdf)),
        "fill": "ask_in_bid_out",
        "stock_exit": "UP trail120bp hold180m smooth_break",
        "option_exit_clock": "synced_to_stock_exit_ts",
        "variants": variants,
        "best": best,
        "note": (
            "2DTE = open_2dte ATM call from lock map (not calendar weeklies). "
            "QQQ filters applied at detect_ts."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# Smooth UP × QQQ × Stock/0DTE/2DTE — May–Jul",
        "",
        f"Entries (capped ≤2/day): **{len(bdf)}**",
        "",
        "## Scoreboard",
        "",
        vdf.to_markdown(index=False),
        "",
        f"## Best (n≥30): `{best}`" if best else "## Best: n/a",
        "",
        "## Notes",
        "",
        "- Option PnL uses **same entry/exit timestamps** as stock trail path (fair vehicle compare).",
        "- `opt_2dte` is trading 2 DTE ATM call (available in open-ladder locks/quotes).",
        "- Ask/bid opponent fills (`FillSpec 1.0/1.0`).",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(vdf.to_string(index=False))
    print("BEST", best)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
