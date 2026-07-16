#!/usr/bin/env python3
"""A/B: day_lock vs open_lock vs open_ladder.

Also writes set_alignment.json explaining only_day/only_lad (usually m5
re-entry path dependence, not missing quotes).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay


def _summary_row(name: str, result: dict) -> dict:
    s = result.get("summary") or result
    return {
        "arm": name,
        "total_ret": s.get("total_ret"),
        "maxdd": s.get("maxdd"),
        "n_trades": s.get("n_trades"),
        "trade_exp": s.get("trade_exp"),
        "trade_win": s.get("trade_win"),
        "contract_mode": s.get("contract_mode"),
        "quote_source_mode": s.get("quote_source_mode"),
        "reentry_mode": s.get("reentry_mode"),
        "only_win_reenter": s.get("only_win_reenter"),
        "position_sizing": s.get("position_sizing"),
        "position_frac": s.get("position_frac"),
        "exit_mode": s.get("exit_mode"),
        "n_size_full": s.get("n_size_full"),
        "n_size_split": s.get("n_size_split"),
    }


def _set_alignment(out_dir: Path) -> dict:
    """Compare trade keys; first entries usually align, later diverge via re-entry."""
    paths = {a: out_dir / a / "trades.csv" for a in ("day_lock", "open_lock", "open_ladder")}
    if not all(p.is_file() for p in paths.values()):
        return {}
    dl = pd.read_csv(paths["day_lock"])
    ld = pd.read_csv(paths["open_ladder"])
    for df in (dl, ld):
        if "direction" not in df.columns and "dir" in df.columns:
            df["direction"] = df["dir"]
        df["date"] = df["date"].astype(str)
    keys = ["date", "symbol", "direction", "n_in_day"]
    dk = set(map(tuple, dl[keys].itertuples(index=False, name=None)))
    lk = set(map(tuple, ld[keys].itertuples(index=False, name=None)))
    only_day = dk - lk
    only_lad = lk - dk
    both = dk & lk

    def _sum_ret(df, keyset):
        if not keyset:
            return {"n": 0, "ret_sum": 0.0, "ret_mean": None}
        mask = df[keys].apply(tuple, axis=1).isin(keyset)
        sub = df.loc[mask]
        return {"n": int(len(sub)), "ret_sum": float(sub["ret"].sum()), "ret_mean": float(sub["ret"].mean())}

    # first-entry alignment (n_in_day==1)
    d1 = dl[dl["n_in_day"] == 1] if "n_in_day" in dl.columns else dl
    l1 = ld[ld["n_in_day"] == 1] if "n_in_day" in ld.columns else ld
    k1 = ["date", "symbol", "direction"]
    s1 = set(map(tuple, d1[k1].itertuples(index=False, name=None)))
    s2 = set(map(tuple, l1[k1].itertuples(index=False, name=None)))

    paired = dl.merge(ld, on=keys, suffixes=("_day", "_lad"), how="inner")
    same_ticker = None
    if not paired.empty and "ticker_day" in paired.columns and "ticker_lad" in paired.columns:
        same_ticker = float((paired["ticker_day"].astype(str) == paired["ticker_lad"].astype(str)).mean())
    elif not paired.empty and "ticker_x" in paired.columns:
        same_ticker = float((paired["ticker_x"].astype(str) == paired["ticker_y"].astype(str)).mean())

    out = {
        "note": (
            "only_day/only_lad under m5_circuit+only_win is mostly re-entry path dependence: "
            "different early P&L → different later n_in_day. With reentry_mode=cooldown_only, "
            "entry clocks align and only_* shrinks; residual gaps are circuit halt / quote misses. "
            "First entries (n_in_day==1) usually overlap; not a missing-quote issue."
        ),
        "n_day": int(len(dl)),
        "n_ladder": int(len(ld)),
        "n_both_keys": int(len(both)),
        "only_day": _sum_ret(dl, only_day),
        "only_ladder": _sum_ret(ld, only_lad),
        "first_entry_overlap": {
            "both": int(len(s1 & s2)),
            "only_day": int(len(s1 - s2)),
            "only_ladder": int(len(s2 - s1)),
        },
        "paired_same_ticker_rate": same_ticker,
    }
    (out_dir / "set_alignment.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"set_alignment": out}, indent=2), flush=True)
    return out


def _apply_reentry_mode(prof: dict, mode: str | None) -> None:
    if not mode:
        return
    trade = prof.setdefault("trade", {})
    trade["reentry_mode"] = mode
    # Keep legacy flag consistent for readers that only look at the bool.
    from maga7.common.reentry import resolve_only_win_reenter

    trade["only_reenter_after_win"] = resolve_only_win_reenter(trade)


def _apply_position_sizing(prof: dict, mode: str | None) -> None:
    if not mode:
        return
    prof.setdefault("trade", {})["position_sizing"] = mode


def _apply_trade_overrides(
    prof: dict,
    *,
    reentry_mode: str | None,
    position_sizing: str | None,
    exit_mode: str | None,
    position_frac: float | None,
) -> None:
    _apply_reentry_mode(prof, reentry_mode)
    _apply_position_sizing(prof, position_sizing)
    trade = prof.setdefault("trade", {})
    if exit_mode:
        trade["exit_mode"] = exit_mode
    if position_frac is not None:
        trade["position_frac"] = float(position_frac)
        trade.setdefault("max_concurrent_positions", 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-profile",
        default=str(ROOT / "maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_lock_research_v1.json"),
    )
    ap.add_argument(
        "--ladder-profile",
        default=str(ROOT / "maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm5otm_v1.json"),
    )
    ap.add_argument("--scheme", default="m5_circuit", choices=["single", "m5", "m5_circuit"])
    ap.add_argument("--quote-source", default="day_iv", choices=["day_iv", "1s", "auto"])
    ap.add_argument(
        "--reentry-mode",
        default=None,
        help="Override trade.reentry_mode on all arms (e.g. cooldown_only, only_win).",
    )
    ap.add_argument(
        "--position-sizing",
        default=None,
        choices=["concurrent", "topk", "live"],
        help="Override trade.position_sizing (concurrent=full sleeve when alone).",
    )
    ap.add_argument(
        "--exit-mode",
        default=None,
        choices=["none", "rails", "mf_flip", "streak_break"],
        help="Override trade.exit_mode (mf_flip = stock mf10 fade exit).",
    )
    ap.add_argument("--position-frac", type=float, default=None, help="Override trade.position_frac.")
    ap.add_argument("--tag", default="open_ladder_ab_dayiv_jan_jul")
    args = ap.parse_args()

    base = load_profile(args.base_profile)
    out_dir = Path(base["_paths"]["results_dir"]) / args.tag
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arms: list[tuple[str, dict]] = []
    common = dict(
        reentry_mode=args.reentry_mode,
        position_sizing=args.position_sizing,
        exit_mode=args.exit_mode,
        position_frac=args.position_frac,
    )

    day = load_profile(args.base_profile)
    day["trade"] = dict(day.get("trade") or {})
    day["trade"]["contract_mode"] = "day_lock"
    day["trade"]["quote_source"] = args.quote_source
    day["trade"].pop("clear_otm_ban_0dte_pct", None)
    _apply_trade_overrides(day, **common)
    if args.quote_source != "day_iv":
        day["_paths"]["quote_1s_root"] = Path("/mnt/s990/data/raw_1s/maga7_mf10_old_lock")
        day.setdefault("paths", {})["quote_1s_root"] = "/mnt/s990/data/raw_1s/maga7_mf10_old_lock"
    arms.append(("day_lock", day))

    ol = load_profile(args.base_profile)
    ol["trade"] = dict(ol.get("trade") or {})
    ol["trade"]["contract_mode"] = "open_lock"
    ol["trade"]["quote_source"] = args.quote_source
    _apply_trade_overrides(ol, **common)
    arms.append(("open_lock", ol))

    ld = load_profile(args.ladder_profile)
    ld["trade"] = dict(ld.get("trade") or {})
    ld["trade"]["contract_mode"] = "open_ladder"
    ld["trade"]["quote_source"] = args.quote_source
    _apply_trade_overrides(ld, **common)
    arms.append(("open_ladder", ld))

    rows = []
    for name, prof in arms:
        print(f"=== {name} ===", flush=True)
        r = run_offline_replay(prof, scheme=args.scheme)
        arm_dir = out_dir / name
        arm_dir.mkdir(parents=True, exist_ok=True)
        (arm_dir / "summary.json").write_text(
            json.dumps(r.get("summary", r), indent=2, default=str), encoding="utf-8"
        )
        trades = r.get("trades")
        if trades is not None and hasattr(trades, "to_csv"):
            trades.to_csv(arm_dir / "trades.csv", index=False)
        rows.append(_summary_row(name, r))
        print(json.dumps(rows[-1], indent=2), flush=True)

    meta = {
        "scheme": args.scheme,
        "quote_source": args.quote_source,
        "reentry_mode": args.reentry_mode,
        "position_sizing": args.position_sizing,
        "exit_mode": args.exit_mode,
        "position_frac": args.position_frac,
        "base_profile": args.base_profile,
        "ladder_profile": args.ladder_profile,
    }
    (out_dir / "ab_summary.json").write_text(
        json.dumps({"meta": meta, "arms": rows}, indent=2), encoding="utf-8"
    )
    _set_alignment(out_dir)
    print("wrote", out_dir / "ab_summary.json", flush=True)



if __name__ == "__main__":
    main()
