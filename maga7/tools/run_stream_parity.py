#!/usr/bin/env python3
"""Stream vs offline parity for maga7 mf10 Top2.

Default stock source for causal evidence is ``stock_1s``
(``/mnt/s990/data/raw_1s/stocks``). ``cache_1m`` (spnq_train) is research-only.
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
from maga7.common.provenance import code_fingerprint
from maga7.common.replay import run_offline_replay
from maga7.common.stock_1s import (
    build_stock_by_from_1s,
    coverage_report,
    regime_gate_from_1s,
    session_dates,
)
from maga7.common.stream_engine import run_stream_replay

PEER3 = (
    ROOT
    / "maga7"
    / "CONFIG"
    / "strategy_profiles"
    / "single_qqq_open_ladder_atm5otm_t30_rails_p20_googl_peer3_v1.json"
)


def _trade_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["key"] = []
        return out
    out["sig_ts"] = pd.to_datetime(out["sig_ts"], utc=True)
    out["key"] = (
        out["date"].astype(str)
        + "|"
        + out["symbol"].astype(str)
        + "|"
        + out["dir"].astype(str)
        + "|"
        + out["n_in_day"].astype(int).astype(str)
    )
    return out


def compare_trades(off: pd.DataFrame, st: pd.DataFrame) -> dict:
    if off.empty and st.empty:
        return {
            "n_offline": 0,
            "n_stream": 0,
            "matched": 0,
            "only_offline": 0,
            "only_stream": 0,
            "ret_max_abs_diff": 0.0,
        }
    a = _trade_key(off)
    b = _trade_key(st)
    merged = a.merge(b, on="key", how="outer", suffixes=("_off", "_st"), indicator=True)
    both = merged[merged["_merge"] == "both"]
    ret_diff = None
    size_diff = None
    reason_mismatch = None
    only_off_keys: list[str] = []
    only_st_keys: list[str] = []
    if len(both):
        ret_diff = float((both["ret_off"] - both["ret_st"]).abs().max())
        if "size_frac_off" in both.columns and "size_frac_st" in both.columns:
            size_diff = float((both["size_frac_off"] - both["size_frac_st"]).abs().max())
        if "reason_off" in both.columns and "reason_st" in both.columns:
            reason_mismatch = int(
                (both["reason_off"].astype(str) != both["reason_st"].astype(str)).sum()
            )
    if len(merged):
        only_off_keys = merged.loc[merged["_merge"] == "left_only", "key"].astype(str).tolist()
        only_st_keys = merged.loc[merged["_merge"] == "right_only", "key"].astype(str).tolist()
    return {
        "n_offline": int(len(a)),
        "n_stream": int(len(b)),
        "matched": int(len(both)),
        "only_offline": int((merged["_merge"] == "left_only").sum()) if len(merged) else 0,
        "only_stream": int((merged["_merge"] == "right_only").sum()) if len(merged) else 0,
        "only_offline_keys": only_off_keys[:50],
        "only_stream_keys": only_st_keys[:50],
        "ret_max_abs_diff": ret_diff,
        "size_frac_max_abs_diff": size_diff,
        "reason_mismatch": reason_mismatch,
        "equity_off": None,
        "equity_st": None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="maga7 stream vs offline parity")
    p.add_argument("--profile", default=str(PEER3))
    p.add_argument("--scheme", default="single", choices=["single", "m5", "m5_circuit"])
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument(
        "--stock-source",
        default="stock_1s",
        choices=["stock_1s", "cache_1m"],
        help="stock_1s = /mnt/s990/data/raw_1s/stocks (required for causal parity); "
        "cache_1m = spnq_train research cache only",
    )
    args = p.parse_args()

    profile = load_profile(args.profile)
    fingerprint = code_fingerprint(profile["_profile_path"])
    if args.start_date:
        profile["date_range"]["start"] = args.start_date
    if args.end_date:
        profile["date_range"]["end"] = args.end_date

    start = profile["date_range"]["start"]
    end = profile["date_range"]["end"]
    tag = args.tag or f"parity_{args.scheme}_{args.stock_source}_{start}_{end}"
    out_dir = Path(profile["_paths"]["results_dir"]) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    stock_by = None
    regime_gate = None
    cov = None
    if args.stock_source == "stock_1s":
        stock_1s = Path(profile["_paths"]["stock_1s_root"])
        if not stock_1s.is_dir():
            raise SystemExit(f"stock_1s_root missing: {stock_1s}")
        dates = session_dates(start, end)
        print(
            f"building stock frames from 1s only: {stock_1s} ({start}..{end}, {len(dates)} sessions)",
            flush=True,
        )
        stock_by = build_stock_by_from_1s(profile, dates=dates, include_refs=True)
        regime_gate = regime_gate_from_1s(profile, stock_by)
        cov = coverage_report(
            stock_by,
            dates=dates,
            symbols=list(profile["symbols"]) + ["QQQ", "VIXY"],
        )
        (out_dir / "stock_1s_coverage.json").write_text(
            json.dumps(cov, indent=2), encoding="utf-8"
        )
        print(json.dumps(cov, indent=2), flush=True)
        # Guard: strategy symbols must have at least some 1s days
        missing_all = [
            s
            for s in profile["symbols"]
            if cov["symbols"].get(s, {}).get("n_days", 0) == 0
        ]
        if missing_all:
            raise SystemExit(f"no stock 1s coverage for symbols: {missing_all}")
    else:
        print(
            "WARNING: stock-source=cache_1m uses spnq_train research cache — "
            "not causal stock-1s parity evidence",
            flush=True,
        )

    # Separate regime gates: Watchdog overlays mutate cfg in-place; sharing one
    # gate lets the last offline day (e.g. HALT) poison the stream baseline snap.
    print("running offline...", flush=True)
    rg_off = (
        regime_gate_from_1s(profile, stock_by)
        if args.stock_source == "stock_1s" and stock_by is not None
        else regime_gate
    )
    off = run_offline_replay(
        profile,
        scheme=args.scheme,
        stock_by=stock_by,
        regime_gate=rg_off,
    )
    print("running stream...", flush=True)
    rg_st = (
        regime_gate_from_1s(profile, stock_by)
        if args.stock_source == "stock_1s" and stock_by is not None
        else regime_gate
    )
    st = run_stream_replay(
        profile,
        scheme=args.scheme,
        stock_by=stock_by,
        regime_gate=rg_st,
    )

    cmp_ = compare_trades(off["trades"], st["trades"])
    cmp_["equity_off"] = off["summary"].get("end_equity")
    cmp_["equity_st"] = st["summary"].get("end_equity")
    cmp_["total_ret_off"] = off["summary"].get("total_ret")
    cmp_["total_ret_st"] = st["summary"].get("total_ret")
    cmp_["strategy_fingerprint"] = fingerprint
    cmp_["stock_source"] = args.stock_source
    cmp_["stock_1s_root"] = str(profile["_paths"].get("stock_1s_root"))
    cmp_["quote_1s_root"] = str(profile["_paths"].get("quote_1s_root"))
    cmp_["profile"] = profile.get("profile_id") or profile.get("profile")
    cmp_["period"] = f"{start}..{end}"
    cmp_["scheme"] = args.scheme
    if cov is not None:
        cmp_["stock_1s_coverage"] = cov
    off["summary"]["strategy_fingerprint"] = fingerprint
    st["summary"]["strategy_fingerprint"] = fingerprint
    off["summary"]["stock_source"] = args.stock_source
    st["summary"]["stock_source"] = args.stock_source
    cmp_["ok"] = (
        cmp_["only_offline"] == 0
        and cmp_["only_stream"] == 0
        and (cmp_["ret_max_abs_diff"] is None or cmp_["ret_max_abs_diff"] < 1e-9)
        and (
            cmp_.get("size_frac_max_abs_diff") is None
            or cmp_["size_frac_max_abs_diff"] < 1e-9
        )
        and (cmp_.get("reason_mismatch") in (None, 0))
    )

    (out_dir / "parity_summary.json").write_text(json.dumps(cmp_, indent=2), encoding="utf-8")
    (out_dir / "offline_summary.json").write_text(
        json.dumps(off["summary"], indent=2), encoding="utf-8"
    )
    (out_dir / "stream_summary.json").write_text(
        json.dumps(st["summary"], indent=2), encoding="utf-8"
    )
    off["trades"].to_csv(out_dir / "trades_offline.csv", index=False)
    st["trades"].to_csv(out_dir / "trades_stream.csv", index=False)
    print(json.dumps({k: v for k, v in cmp_.items() if k != "stock_1s_coverage"}, indent=2))
    print(f"wrote {out_dir}")
    if not cmp_["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
