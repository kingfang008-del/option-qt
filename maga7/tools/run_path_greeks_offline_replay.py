#!/usr/bin/env python3
"""Offline multi-day L2 path-Greeks replay (no Redis).

Loads Mag7 open-ladder 1s quotes + stock 1s, simulates ``path_greeks_exit``
presets on existing trade books (e.g. May–Jun baseline_t30 / wash_and_opt).

L2 can only fire *earlier* than the book's exit (stacking semantics).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.path_greeks_exit import PathGreeksState, cfg_from_preset

NY = "America/New_York"
QUOTE_ROOT = Path("/mnt/s990/data/raw_1s/maga7_mf10_open_ladder_otm5")
STOCK_ROOT = Path("/mnt/s990/data/raw_1s/stocks")
DEFAULT_BOOKS = {
    "baseline_t30": Path(
        "/mnt/s990/data/maga7/results/path_hold_opt_chop_may_jun_v1/baseline_t30/trades.csv"
    ),
    "wash_and_opt": Path(
        "/mnt/s990/data/maga7/results/path_hold_opt_chop_may_jun_v1/wash_and_opt/trades.csv"
    ),
    "lit_always": Path(
        "/mnt/s990/data/maga7/results/path_hold_opt_chop_may_jun_v1/lit_always/trades.csv"
    ),
}


def _parse_occ(ticker: str):
    s = str(ticker).replace(" ", "")
    m = re.match(r"^([A-Z]+)(\d{6})([CP])(\d{8})$", s)
    if not m:
        raise ValueError(ticker)
    sym, ymd, cp, strike_s = m.groups()
    return sym, ymd, float(strike_s) / 1000.0, cp.lower()


_OPT_DAY: dict[tuple[str, str], pd.DataFrame] = {}
_STK_DAY: dict[tuple[str, str], pd.DataFrame] = {}


def _load_option_day(sym: str, date: str) -> pd.DataFrame:
    key = (sym, date)
    if key in _OPT_DAY:
        return _OPT_DAY[key]
    p = QUOTE_ROOT / sym / f"{sym}_{date}.parquet"
    if not p.is_file():
        _OPT_DAY[key] = pd.DataFrame()
        return _OPT_DAY[key]
    df = pd.read_parquet(p)
    df = df.copy()
    df["_tkey"] = df["ticker"].astype(str).str.replace(" ", "", regex=False)
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    try:
        ts = ts.dt.tz_convert(NY)
    except Exception:
        pass
    df["ts"] = ts
    mid = pd.to_numeric(df.get("mid_price"), errors="coerce")
    if mid.isna().all():
        bid = pd.to_numeric(df["bid"], errors="coerce")
        ask = pd.to_numeric(df["ask"], errors="coerce")
        mid = (bid + ask) / 2.0
    df["mid"] = mid
    _OPT_DAY[key] = df
    return df


def _load_option_path(sym: str, date: str, ticker: str) -> pd.DataFrame:
    df = _load_option_day(sym, date)
    if df.empty:
        return df
    want = str(ticker).replace(" ", "")
    sub = df.loc[df["_tkey"] == want, ["ts", "mid"]].copy()
    if sub.empty:
        suf = want[-15:]
        sub = df.loc[df["_tkey"].str.endswith(suf), ["ts", "mid"]].copy()
    if sub.empty:
        return pd.DataFrame()
    return sub.dropna().sort_values("ts").drop_duplicates("ts").reset_index(drop=True)


def _load_stock_path(sym: str, date: str) -> pd.DataFrame:
    key = (sym, date)
    if key in _STK_DAY:
        return _STK_DAY[key]
    p = STOCK_ROOT / sym / f"{sym}_{date}.parquet"
    if not p.is_file():
        _STK_DAY[key] = pd.DataFrame()
        return _STK_DAY[key]
    df = pd.read_parquet(p)
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    try:
        ts = ts.dt.tz_convert(NY)
    except Exception:
        pass
    close = pd.to_numeric(df["close"], errors="coerce")
    out = pd.DataFrame({"ts": ts, "S": close}).dropna().sort_values("ts")
    _STK_DAY[key] = out.drop_duplicates("ts").reset_index(drop=True)
    return _STK_DAY[key]


def simulate_trade(tr: dict, preset: str) -> dict:
    cfg, naive = cfg_from_preset(preset)
    ticker = str(tr.get("ticker") or tr.get("contract") or "")
    sym = str(tr["symbol"]).upper()
    date = str(tr["date"])[:10]
    try:
        _, ymd, K, cp = _parse_occ(ticker)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"bad_ticker:{exc}",
            "final_ret": float(tr["ret"]),
            "base_ret": float(tr["ret"]),
            "lift": 0.0,
        }
    entry_ts = pd.Timestamp(tr["entry_ts"])
    if entry_ts.tzinfo is None:
        entry_ts = entry_ts.tz_localize(NY)
    else:
        entry_ts = entry_ts.tz_convert(NY)
    exit_ts = pd.Timestamp(tr["exit_ts"])
    if exit_ts.tzinfo is None:
        exit_ts = exit_ts.tz_localize(NY)
    else:
        exit_ts = exit_ts.tz_convert(NY)

    opt = _load_option_path(sym, date, ticker)
    stk = _load_stock_path(sym, date)
    if opt.empty or stk.empty:
        return {
            "ok": False,
            "error": "no_path",
            "final_ret": float(tr["ret"]),
            "base_ret": float(tr["ret"]),
            "lift": 0.0,
            "n_opt": int(len(opt)),
            "n_stk": int(len(stk)),
        }
    # merge asof stock onto option
    path = pd.merge_asof(
        opt.sort_values("ts"),
        stk.sort_values("ts"),
        on="ts",
        direction="backward",
    )
    path = path.loc[(path["ts"] >= entry_ts) & (path["ts"] <= exit_ts)].dropna(
        subset=["mid", "S"]
    )
    if path.empty:
        return {
            "ok": False,
            "error": "empty_hold_path",
            "final_ret": float(tr["ret"]),
            "base_ret": float(tr["ret"]),
            "lift": 0.0,
        }

    expiry = pd.Timestamp(
        f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:6]}", tz=NY
    ) + pd.Timedelta(hours=16)
    st = PathGreeksState(
        entry_px=float(tr["entry"]),
        K=K,
        cp=cp,
        expiry_ts=float(expiry.timestamp()),
        cfg=cfg,
        naive_half_peak=naive,
        entry_ts=float(entry_ts.timestamp()),
    )
    l2_reason = None
    l2_ret = None
    l2_hold = None
    for _, row in path.iterrows():
        reason, met = st.on_tick(
            ts=float(row["ts"].timestamp()),
            mid=float(row["mid"]),
            S=float(row["S"]),
        )
        if reason:
            l2_reason = reason
            l2_ret = float(met["opt_ret"])
            l2_hold = float((row["ts"] - entry_ts).total_seconds())
            break
    base = float(tr["ret"])
    final = l2_ret if l2_ret is not None else base
    return {
        "ok": True,
        "date": date,
        "symbol": sym,
        "ticker": ticker,
        "base_ret": base,
        "base_reason": str(tr.get("reason")),
        "final_ret": final,
        "l2_reason": l2_reason,
        "l2_hold_sec": l2_hold,
        "lift": float(final - base),
        "mfe": float(st.peak_ret),
        "n_path": int(len(path)),
        "preset": preset,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--books",
        default="baseline_t30,wash_and_opt,lit_always",
        help="comma names from default map or path=...",
    )
    ap.add_argument("--presets", default="off,naive,winner_safe,toxic_only")
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/path_greeks_offline_may_jun_v1",
    )
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    book_items = []
    for tok in args.books.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" in tok:
            name, path = tok.split("=", 1)
            book_items.append((name, Path(path)))
        elif tok in DEFAULT_BOOKS:
            book_items.append((tok, DEFAULT_BOOKS[tok]))
        else:
            raise SystemExit(f"unknown book {tok}")

    presets = [x.strip() for x in args.presets.split(",") if x.strip()]
    details = []
    score = []
    for bname, bpath in book_items:
        trades = pd.read_csv(bpath)
        print(f"=== book {bname} n={len(trades)} ===", flush=True)
        for preset in presets:
            rows = []
            for _, tr in trades.iterrows():
                rows.append(simulate_trade(tr.to_dict(), preset))
            df = pd.DataFrame(rows)
            df["book"] = bname
            df["preset"] = preset
            details.append(df)
            ok = df[df["ok"] == True]  # noqa: E712
            score.append(
                {
                    "book": bname,
                    "preset": preset,
                    "n": int(len(df)),
                    "n_ok": int(len(ok)),
                    "n_l2": int(ok["l2_reason"].notna().sum()) if len(ok) else 0,
                    "sum_base": float(ok["base_ret"].sum()) if len(ok) else None,
                    "sum_final": float(ok["final_ret"].sum()) if len(ok) else None,
                    "sum_lift": float(ok["lift"].sum()) if len(ok) else None,
                    "n_no_path": int((df.get("error") == "no_path").sum())
                    if "error" in df.columns
                    else 0,
                }
            )
            s = score[-1]
            print(
                f"  {preset:12s} final={s['sum_final']:+.3f} base={s['sum_base']:+.3f} "
                f"lift={s['sum_lift']:+.3f} l2={s['n_l2']}/{s['n_ok']} miss={s['n_no_path']}",
                flush=True,
            )

    detail = pd.concat(details, ignore_index=True)
    detail.to_csv(out / "trades_detail.csv", index=False)
    sb = pd.DataFrame(score)
    sb.to_csv(out / "scoreboard.csv", index=False)
    (out / "scoreboard.json").write_text(
        json.dumps(score, indent=2), encoding="utf-8"
    )
    print(sb.to_string(index=False))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
