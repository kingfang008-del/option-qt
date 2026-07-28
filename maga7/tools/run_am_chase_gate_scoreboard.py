#!/usr/bin/env python3
"""May–Jul AM A+B chase/liquidity entry-gate scoreboard.

Baseline books (locked research):
  A = morning both_fo08
  B = segB both08_ca_up_only

Arms (post-hoc on filled books; FO/MF/mid causal asof entry):
  CTRL0
  FO_MAX_010 / FO_MAX_015          — drop if true |fav_from_open| > thr
  MF1_SAME / MF10_SAME             — drop if money-flow against dir
  CA_DN                            — swap B DN outcomes to both08_ca book
  MID_{075,100,125}_BLOCK          — drop if entry_mid < thr
  MID_{075,100,125}_HALF           — size×0.5 (pnl×0.5) if entry_mid < thr
  Combos of the stronger single arms

Also reports whether each arm would block the 2026-07-27 live NVDA chase
(true FO≈3.36%, mid≈0.93, mf10 against).

Example:
  PYTHONPATH=. python -m maga7.tools.run_am_chase_gate_scoreboard \\
    --tag research_am_chase_gate_20260728
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.fills import FillSpec
from maga7.common.replay import load_quotes, month_list, path_for_ticker, to_ny
from maga7.common.signals import attach_mf_features, load_stock_month_files

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
A_BOOK = (
    "/mnt/s990/data/maga7/results/research_am_morning_both_accept_20260728/"
    "book_both_fo08.csv"
)
B_BOOK = (
    "/mnt/s990/data/maga7/results/research_am_segB_both_ddctrl_20260728/"
    "book_both08_ca_up_only.csv"
)
B_CA_BOTH = (
    "/mnt/s990/data/maga7/results/research_am_segB_both_ddctrl_20260728/"
    "book_both08_ca.csv"
)

# Live 2026-07-27 NVDA AM pulse DN loss probe
LIVE_PROBE = {
    "date": "2026-07-27",
    "symbol": "NVDA",
    "dir": "DN",
    "true_fo": 0.033603,
    "entry_mid": 0.93,
    "mf1": None,  # sparse; mf10 known against
    "mf10": 1.0,  # positive while DN → against
}


def _stats(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {
            "n": 0,
            "win": None,
            "add": 0.0,
            "compound": 0.0,
            "mult": 1.0,
            "maxdd": 0.0,
            "n_a": 0,
            "n_b": 0,
            "add_a": 0.0,
            "add_b": 0.0,
        }
    d = df.groupby("date")["pnl_frac"].sum().sort_index()
    eq = (1.0 + d).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    a = df[df["pool"] == "A"]
    b = df[df["pool"] == "B"]
    return {
        "n": int(len(df)),
        "win": float((df["ret"] > 0).mean()),
        "add": float(d.sum()),
        "compound": float(eq.iloc[-1] - 1.0),
        "mult": float(eq.iloc[-1]),
        "maxdd": float(dd.min()) if len(dd) else 0.0,
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "add_a": float(a["pnl_frac"].sum()) if len(a) else 0.0,
        "add_b": float(b["pnl_frac"].sum()) if len(b) else 0.0,
    }


def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    out = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    return out


def _annotate(
    book: pd.DataFrame,
    *,
    stock_root: Path,
    quote_root: Path,
    months: list[str],
) -> pd.DataFrame:
    fill = FillSpec(entry_frac=0.75, exit_frac=0.75)
    stock_cache: dict[str, pd.DataFrame] = {}
    quote_cache: dict[tuple[str, str], pd.DataFrame | None] = {}
    rows: list[dict[str, Any]] = []

    for _, r in book.iterrows():
        sym = str(r["symbol"]).upper()
        date = str(r["date"])
        ts = to_ny(r["entry_ts"])
        d = str(r["dir"]).upper()
        row = dict(r)
        row["date"] = date
        row["symbol"] = sym
        row["dir"] = d

        if sym not in stock_cache:
            sdf = load_stock_month_files(stock_root, sym, months)
            if sdf is None or sdf.empty:
                stock_cache[sym] = pd.DataFrame()
            else:
                stock_cache[sym] = attach_mf_features(sdf)
        sdf = stock_cache[sym]
        true_fo = np.nan
        mf1 = np.nan
        mf10 = np.nan
        if sdf is not None and not sdf.empty:
            day = sdf[sdf["date"].astype(str) == date]
            if not day.empty:
                day_open = float(day.iloc[0]["open"])
                up = day[pd.to_datetime(day["timestamp"]) <= ts]
                if not up.empty and day_open > 0:
                    px = float(up.iloc[-1]["close"])
                    signed = (px / day_open - 1.0) if d == "UP" else (day_open - px) / day_open
                    true_fo = float(signed)
                    # mf1 = last bar net$ (1m); mf10 from features
                    bar = up.iloc[-1]
                    if "net$" in up.columns:
                        mf1 = float(bar["net$"]) if pd.notna(bar["net$"]) else np.nan
                    elif "mf_fast" in up.columns and pd.notna(bar.get("mf_fast")):
                        # fallback if net$ stripped; rebuild quickly
                        mf1 = np.nan
                    if "mf10" in up.columns and pd.notna(bar.get("mf10")):
                        mf10 = float(bar["mf10"])
                    # Ensure mf1 from net$ path: recompute if missing
                    if not np.isfinite(mf1) and {"high", "low", "close", "volume"}.issubset(up.columns):
                        last = up.iloc[-1]
                        hl = float(last["high"]) - float(last["low"])
                        if hl <= 0:
                            hl = np.nan
                        buy = (
                            ((float(last["close"]) - float(last["low"])) / hl)
                            if np.isfinite(hl)
                            else 0.5
                        ) * float(last["volume"])
                        sell = (
                            ((float(last["high"]) - float(last["close"])) / hl)
                            if np.isfinite(hl)
                            else 0.5
                        ) * float(last["volume"])
                        mf1 = (buy - sell) * float(last["close"])

        row["true_fo"] = true_fo
        row["mf1"] = mf1
        row["mf10"] = mf10

        qkey = (sym, date)
        if qkey not in quote_cache:
            quote_cache[qkey] = load_quotes(quote_root, sym, date)
        path = path_for_ticker(quote_cache[qkey], str(r["ticker"]))
        mid = bid = ask = spr = np.nan
        if path is not None and not path.empty:
            upq = path[path["timestamp"] <= ts]
            if not upq.empty:
                q = upq.iloc[-1]
                bid = float(q["bid"])
                ask = float(q["ask"])
                mid = 0.5 * (bid + ask)
                spr = (ask - bid) / mid if mid > 0 else np.nan
                _ = fill.buy(bid, ask)
        row["entry_mid"] = mid
        row["spread_pct"] = spr
        rows.append(row)
    return pd.DataFrame(rows)


def _mf_against(dir_: str, mf: float) -> bool:
    if not np.isfinite(mf):
        return False  # missing → allow (no false block)
    if dir_ == "UP":
        return float(mf) < 0
    return float(mf) > 0


def _live_action(apply_fn: Callable[[pd.DataFrame], pd.DataFrame]) -> str:
    """block | half | pass for the 2026-07-27 NVDA probe."""
    probe = pd.DataFrame(
        [
            {
                "date": LIVE_PROBE["date"],
                "symbol": LIVE_PROBE["symbol"],
                "dir": LIVE_PROBE["dir"],
                "pool": "A",
                "ret": -0.214,
                "size": 0.1,
                "pnl_frac": -0.0214,
                "true_fo": LIVE_PROBE["true_fo"],
                "entry_mid": LIVE_PROBE["entry_mid"],
                "mf1": np.nan,
                "mf10": LIVE_PROBE["mf10"],
                "ticker": "NVDA260727P00200000",
                "entry_ts": "2026-07-27 10:13:00-04:00",
                "exit_reason": "sl",
            }
        ]
    )
    out = apply_fn(probe)
    if out is None or out.empty:
        return "block"
    size = float(out.iloc[0]["size"])
    if size <= 1e-12:
        return "block"
    if size < 0.1 - 1e-12:
        return "half"
    return "pass"

def _apply_fo_max(df: pd.DataFrame, thr: float, pools: set[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    pools = pools or {"A", "B"}
    mask = out["pool"].isin(pools) & out["true_fo"].notna() & (out["true_fo"] > thr)
    return out.loc[~mask].copy()


def _apply_mf_same(df: pd.DataFrame, col: str, pools: set[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    pools = pools or {"A", "B"}
    keep = []
    for _, r in out.iterrows():
        if r["pool"] not in pools:
            keep.append(True)
            continue
        keep.append(not _mf_against(str(r["dir"]), float(r[col]) if pd.notna(r[col]) else np.nan))
    return out.loc[keep].copy()


def _apply_mid_block(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    out = df.copy()
    return out[out["entry_mid"].isna() | (out["entry_mid"] >= thr)].copy()


def _apply_mid_half(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    out = df.copy()
    cheap = out["entry_mid"].notna() & (out["entry_mid"] < thr)
    out.loc[cheap, "size"] = out.loc[cheap, "size"].astype(float) * 0.5
    out.loc[cheap, "pnl_frac"] = out.loc[cheap, "ret"].astype(float) * out.loc[cheap, "size"].astype(
        float
    )
    return out


def _apply_ca_dn(base: pd.DataFrame, ca_both: pd.DataFrame) -> pd.DataFrame:
    """Replace B DN rows with CA-both outcomes when keys match."""
    keys = ["date", "symbol", "dir", "entry_ts", "ticker"]
    a = base[base["pool"] == "A"].copy()
    b = base[base["pool"] == "B"].copy()
    b_up = b[b["dir"] == "UP"].copy()
    b_dn = b[b["dir"] == "DN"].copy()
    ca = ca_both.copy()
    ca["date"] = ca["date"].astype(str)
    ca_dn = ca[ca["dir"].astype(str).str.upper() == "DN"][keys + ["ret", "exit_reason", "hold_sec", "size", "pnl_frac"]]
    merged = b_dn.merge(ca_dn, on=keys, how="left", suffixes=("", "_ca"))
    use = merged["ret_ca"].notna()
    merged.loc[use, "ret"] = merged.loc[use, "ret_ca"]
    merged.loc[use, "exit_reason"] = merged.loc[use, "exit_reason_ca"]
    if "hold_sec_ca" in merged.columns:
        merged.loc[use, "hold_sec"] = merged.loc[use, "hold_sec_ca"]
    # keep original size; recompute pnl
    merged["pnl_frac"] = merged["ret"].astype(float) * merged["size"].astype(float)
    drop_cols = [c for c in merged.columns if c.endswith("_ca")]
    b_dn2 = merged.drop(columns=drop_cols)
    return pd.concat([a, b_up, b_dn2], ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="research_am_chase_gate_20260728")
    ap.add_argument("--a-book", default=A_BOOK)
    ap.add_argument("--b-book", default=B_BOOK)
    ap.add_argument("--b-ca-book", default=B_CA_BOTH)
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--end", default="2026-07-23")
    ap.add_argument("--feb-start", default="2026-02-01")
    ap.add_argument("--feb-end", default="2026-03-31")
    ap.add_argument("--min-retain", type=float, default=0.85)
    args = ap.parse_args(argv)

    prof = load_profile(PROFILE)
    paths = prof["_paths"]
    stock_root = Path(paths["stock_root"])
    quote_root = Path(paths["quote_1s_root"])
    out_dir = Path(paths["results_dir"]) / str(args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    a = pd.read_csv(args.a_book)
    b = pd.read_csv(args.b_book)
    b_ca = pd.read_csv(args.b_ca_book)
    a["date"] = a["date"].astype(str)
    b["date"] = b["date"].astype(str)
    b_ca["date"] = b_ca["date"].astype(str)
    a["pool"] = "A"
    b["pool"] = "B"
    base = pd.concat([a, b], ignore_index=True)

    months = month_list(args.feb_start, args.end)
    print(f"[annotate] n={len(base)} months={months}", flush=True)
    ann = _annotate(base, stock_root=stock_root, quote_root=quote_root, months=months)
    ann.to_csv(out_dir / "annotated_base.csv", index=False)

    # Precompute CA_DN full annotated by swapping B DN then reusing mid/fo from base keys
    ca_dn_base = _apply_ca_dn(ann, b_ca)
    # carry annotations from ann via merge
    meta_cols = ["true_fo", "mf1", "mf10", "entry_mid", "spread_pct"]
    keys = ["date", "symbol", "dir", "entry_ts", "ticker", "pool"]
    ca_ann = ca_dn_base.merge(
        ann[keys + meta_cols],
        on=keys,
        how="left",
        suffixes=("", "_x"),
    )
    for c in meta_cols:
        if f"{c}_x" in ca_ann.columns:
            ca_ann[c] = ca_ann[c].combine_first(ca_ann[f"{c}_x"])
            ca_ann = ca_ann.drop(columns=[f"{c}_x"])

    def compose(*fns: Callable[[pd.DataFrame], pd.DataFrame]) -> Callable[[pd.DataFrame], pd.DataFrame]:
        def _fn(df: pd.DataFrame) -> pd.DataFrame:
            out = df
            for f in fns:
                out = f(out)
            return out

        return _fn

    arms: list[tuple[str, Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame]] = [
        ("CTRL0", lambda df: df.copy(), ann),
        ("FO_MAX_010", lambda df: _apply_fo_max(df, 0.01), ann),
        ("FO_MAX_015", lambda df: _apply_fo_max(df, 0.015), ann),
        ("FO_MAX_010_A", lambda df: _apply_fo_max(df, 0.01, {"A"}), ann),
        ("FO_MAX_015_A", lambda df: _apply_fo_max(df, 0.015, {"A"}), ann),
        ("MF1_SAME", lambda df: _apply_mf_same(df, "mf1"), ann),
        ("MF10_SAME", lambda df: _apply_mf_same(df, "mf10"), ann),
        ("CA_DN", lambda df: df.copy(), ca_ann),
        ("MID_075_BLOCK", lambda df: _apply_mid_block(df, 0.75), ann),
        ("MID_100_BLOCK", lambda df: _apply_mid_block(df, 1.00), ann),
        ("MID_125_BLOCK", lambda df: _apply_mid_block(df, 1.25), ann),
        ("MID_075_HALF", lambda df: _apply_mid_half(df, 0.75), ann),
        ("MID_100_HALF", lambda df: _apply_mid_half(df, 1.00), ann),
        ("MID_125_HALF", lambda df: _apply_mid_half(df, 1.25), ann),
        (
            "FO015_MID100_HALF",
            compose(lambda df: _apply_fo_max(df, 0.015), lambda df: _apply_mid_half(df, 1.00)),
            ann,
        ),
        (
            "FO010_MID100_BLOCK",
            compose(lambda df: _apply_fo_max(df, 0.01), lambda df: _apply_mid_block(df, 1.00)),
            ann,
        ),
        (
            "FO015_MF1",
            compose(lambda df: _apply_fo_max(df, 0.015), lambda df: _apply_mf_same(df, "mf1")),
            ann,
        ),
        (
            "FO015_A_MID100_HALF",
            compose(lambda df: _apply_fo_max(df, 0.015, {"A"}), lambda df: _apply_mid_half(df, 1.00)),
            ann,
        ),
    ]

    ctrl_may = _stats(_slice(ann, args.start, args.end))
    rows_out: list[dict[str, Any]] = []
    for name, fn, src in arms:
        may = _stats(_slice(fn(src), args.start, args.end))
        feb = _stats(_slice(fn(src), args.feb_start, args.feb_end))
        retain = (
            float(may["mult"] / ctrl_may["mult"])
            if ctrl_may["mult"] and ctrl_may["mult"] > 0
            else None
        )
        live_act = _live_action(fn)
        live_blk = live_act == "block"
        dd_ok = may["maxdd"] >= ctrl_may["maxdd"] - 1e-9  # not worse (less negative ok)
        verdict = "PASS" if (
            retain is not None
            and retain + 1e-12 >= float(args.min_retain)
            and live_blk
            and (may["maxdd"] >= -0.20)
        ) else ("BASE" if name == "CTRL0" else "FAIL")
        if name == "CTRL0":
            verdict = "BASE"
        # Soft promote: half-size mitigates live + strong retain
        if verdict == "FAIL" and live_act == "half" and retain is not None and retain >= float(args.min_retain):
            verdict = "SOFT"
        rows_out.append(
            {
                "name": name,
                "live_nvda": live_act,
                "live_nvda_blocked": live_blk,
                "may_n": may["n"],
                "may_win": may["win"],
                "may_add": may["add"],
                "may_mult": may["mult"],
                "may_maxdd": may["maxdd"],
                "may_add_a": may["add_a"],
                "may_add_b": may["add_b"],
                "mult_retain": retain,
                "feb_n": feb["n"],
                "feb_add": feb["add"],
                "feb_win": feb["win"],
                "feb_maxdd": feb["maxdd"],
                "dd_vs_ctrl": float(may["maxdd"] - ctrl_may["maxdd"]),
                "verdict": verdict,
            }
        )
        print(
            f"[{name}] may_mult={may['mult']:.3f} retain={retain} "
            f"maxdd={may['maxdd']:.3%} live={live_act} → {verdict}",
            flush=True,
        )

    sb = pd.DataFrame(rows_out).sort_values(
        by=["live_nvda_blocked", "mult_retain", "may_maxdd"],
        ascending=[False, False, False],
    )
    sb.to_csv(out_dir / "scoreboard.csv", index=False)
    summary = {
        "tag": args.tag,
        "window_may": [args.start, args.end],
        "window_feb": [args.feb_start, args.feb_end],
        "ctrl_may": ctrl_may,
        "live_probe": LIVE_PROBE,
        "min_retain": args.min_retain,
        "note": (
            "day_open RTH latch fixed in am_pulse_scout; this board is post-hoc "
            "entry filters on locked A/B books. Prefer PASS with live block + retain≥min."
        ),
        "scoreboard": rows_out,
        "promote": [r["name"] for r in rows_out if r["verdict"] == "PASS"],
        "soft_promote": [r["name"] for r in rows_out if r["verdict"] == "SOFT"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out_dir), "promote": summary["promote"], "soft": summary["soft_promote"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
