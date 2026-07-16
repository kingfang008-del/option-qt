#!/usr/bin/env python3
"""Stream vs offline parity for maga7 mf10 Top2."""
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
from maga7.common.replay import run_offline_replay
from maga7.common.stream_engine import run_stream_replay


def _trade_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
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
        return {"n_offline": 0, "n_stream": 0, "matched": 0, "only_offline": 0, "only_stream": 0, "ret_max_abs_diff": 0.0}
    a = _trade_key(off)
    b = _trade_key(st)
    merged = a.merge(b, on="key", how="outer", suffixes=("_off", "_st"), indicator=True)
    both = merged[merged["_merge"] == "both"]
    ret_diff = None
    size_diff = None
    reason_mismatch = None
    if len(both):
        ret_diff = float((both["ret_off"] - both["ret_st"]).abs().max())
        if "size_frac_off" in both.columns and "size_frac_st" in both.columns:
            size_diff = float((both["size_frac_off"] - both["size_frac_st"]).abs().max())
        if "reason_off" in both.columns and "reason_st" in both.columns:
            reason_mismatch = int((both["reason_off"].astype(str) != both["reason_st"].astype(str)).sum())
    return {
        "n_offline": int(len(a)),
        "n_stream": int(len(b)),
        "matched": int(len(both)),
        "only_offline": int((merged["_merge"] == "left_only").sum()),
        "only_stream": int((merged["_merge"] == "right_only").sum()),
        "ret_max_abs_diff": ret_diff,
        "size_frac_max_abs_diff": size_diff,
        "reason_mismatch": reason_mismatch,
        "equity_off": None,
        "equity_st": None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="maga7 stream vs offline parity")
    p.add_argument("--profile", default=None)
    p.add_argument("--scheme", default="single", choices=["single", "m5", "m5_circuit"])
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    profile = load_profile(args.profile)
    if args.start_date:
        profile["date_range"]["start"] = args.start_date
    if args.end_date:
        profile["date_range"]["end"] = args.end_date

    out_dir = Path(profile["_paths"]["results_dir"]) / (args.tag or f"parity_{args.scheme}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("running offline...", flush=True)
    off = run_offline_replay(profile, scheme=args.scheme)
    print("running stream...", flush=True)
    st = run_stream_replay(profile, scheme=args.scheme)

    cmp_ = compare_trades(off["trades"], st["trades"])
    cmp_["equity_off"] = off["summary"].get("end_equity")
    cmp_["equity_st"] = st["summary"].get("end_equity")
    cmp_["total_ret_off"] = off["summary"].get("total_ret")
    cmp_["total_ret_st"] = st["summary"].get("total_ret")
    cmp_["ok"] = (
        cmp_["only_offline"] == 0
        and cmp_["only_stream"] == 0
        and (cmp_["ret_max_abs_diff"] is None or cmp_["ret_max_abs_diff"] < 1e-9)
    )

    (out_dir / "parity_summary.json").write_text(json.dumps(cmp_, indent=2), encoding="utf-8")
    (out_dir / "offline_summary.json").write_text(json.dumps(off["summary"], indent=2), encoding="utf-8")
    (out_dir / "stream_summary.json").write_text(json.dumps(st["summary"], indent=2), encoding="utf-8")
    off["trades"].to_csv(out_dir / "trades_offline.csv", index=False)
    st["trades"].to_csv(out_dir / "trades_stream.csv", index=False)
    print(json.dumps(cmp_, indent=2))
    print(f"wrote {out_dir}")
    if not cmp_["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
