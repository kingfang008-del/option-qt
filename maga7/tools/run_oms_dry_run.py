#!/usr/bin/env python3
"""S3 Mag7 OMS dry-run: Scanner (1s→1m) → fill_model limits → 1s quote sim (no IBKR).

Default profile = temp production open_ladder OTM5 + mf_flip p20.
Default scheme = m5_circuit with interleaved fill feedback (only_win).
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

from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.config import load_profile
from maga7.common.replay import month_list, run_offline_replay
from maga7.common.signals import load_stock_month_files
from maga7.live.oms_dry import Mag7OmsDryRunner
from maga7.live.scanner import Mag7Scanner, write_signal_audit

PROD_TEMP = (
    ROOT
    / "maga7"
    / "CONFIG"
    / "strategy_profiles"
    / "m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p20_v1.json"
)


def _dates(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end)]


def _drive_interleaved(
    profile: dict,
    start: str,
    end: str,
    *,
    ingest: str,
    scheme: str,
    runner: Mag7OmsDryRunner,
) -> Mag7Scanner:
    scanner = Mag7Scanner.from_profile(profile, scheme=scheme)
    runner.scanner = scanner
    scanner.on_signal = runner.process_one

    if ingest == "1s":
        stock_1s = profile["_paths"]["stock_1s_root"]
        frames = []
        for date in _dates(start, end):
            for sym in profile["symbols"]:
                raw = load_stock_1s_day(stock_1s, sym, date)
                if raw.empty:
                    continue
                raw = raw.copy()
                raw["symbol"] = sym
                frames.append(raw)
        if not frames:
            raise SystemExit(f"no stock 1s under {stock_1s} for {start}..{end}")
        all_ticks = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"])
        for r in all_ticks.itertuples(index=False):
            scanner.on_stock_second(
                r.symbol,
                {
                    "timestamp": r.timestamp,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                },
            )
        scanner.flush_seconds()
    else:
        months = month_list(start, end)
        frames = []
        for sym in profile["symbols"]:
            raw = load_stock_month_files(profile["_paths"]["stock_root"], sym, months)
            if raw.empty:
                continue
            raw = raw[(raw["date"] >= start) & (raw["date"] <= end)].copy()
            raw["symbol"] = sym
            frames.append(raw)
        if not frames:
            raise SystemExit("no 1m stock bars")
        all_bars = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"])
        for r in all_bars.itertuples(index=False):
            scanner.on_stock_bar(
                r.symbol,
                {
                    "timestamp": r.timestamp,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                },
            )
    return scanner


def main() -> None:
    p = argparse.ArgumentParser(description="Mag7 OMS dry-run (S3)")
    p.add_argument("--profile", default=str(PROD_TEMP))
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", default=None)
    p.add_argument("--ingest", choices=["1s", "1m"], default="1s", help="stock ingest for scanner")
    p.add_argument("--scheme", default="m5_circuit")
    p.add_argument("--tag", default=None, help="results subdir name")
    p.add_argument("--compare-offline", action="store_true", help="also run offline & diff trades")
    args = p.parse_args()

    profile = load_profile(args.profile)
    end = args.end_date or args.start_date
    profile["date_range"]["start"] = args.start_date
    profile["date_range"]["end"] = end

    tag = args.tag or f"oms_dry_{args.scheme}_{args.ingest}_{args.start_date}_{end}"
    out_dir = Path(profile["_paths"]["results_dir"]) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = Mag7OmsDryRunner(profile)
    scanner = _drive_interleaved(
        profile,
        args.start_date,
        end,
        ingest=args.ingest,
        scheme=args.scheme,
        runner=runner,
    )
    write_signal_audit(scanner.signals, out_dir / "signals.jsonl")

    summary = runner.finalize_summary()
    summary["ingest"] = args.ingest
    summary["scheme"] = args.scheme
    summary["start"] = args.start_date
    summary["end"] = end
    summary["n_signals"] = len(scanner.signals)
    summary["profile"] = profile.get("profile_id") or profile.get("profile")
    runner.summary = summary
    runner.write(out_dir)

    print(json.dumps(summary, indent=2))
    print(f"→ {out_dir}")
    for t in runner.trades[:20]:
        print(
            f"  {t.date} rank={t.rank} {t.symbol} {t.direction} "
            f"ret={t.ret:+.1%} size={t.qty_frac:.3f} {t.reason}"
        )

    if args.compare_offline:
        result = run_offline_replay(profile, scheme=args.scheme)
        off_sum = result["summary"]
        (out_dir / "offline_summary.json").write_text(json.dumps(off_sum, indent=2), encoding="utf-8")
        ot = result["trades"].copy()
        ot["date"] = ot["date"].astype(str)
        dry = pd.DataFrame([t.__dict__ for t in runner.trades])

        def _norm_ts(s: pd.Series) -> pd.Series:
            return s.astype(str).str.replace("T", " ", regex=False)

        if "entry_ts" in ot.columns and "entry_ts" in dry.columns:
            ot = ot.copy()
            dry = dry.copy()
            ot["entry_ts"] = _norm_ts(ot["entry_ts"])
            dry["entry_ts"] = _norm_ts(dry["entry_ts"])
            m = dry.merge(
                ot,
                on=["date", "symbol", "entry_ts"],
                how="outer",
                suffixes=("_dry", "_off"),
                indicator=True,
            )
        else:
            m = dry.merge(ot, on=["date", "symbol"], how="outer", suffixes=("_dry", "_off"), indicator=True)
        m.to_csv(out_dir / "compare_offline.csv", index=False)
        both = m[m["_merge"] == "both"]
        delta = None
        if len(both) and "ret_dry" in both.columns and "ret_off" in both.columns:
            delta = (both["ret_dry"] - both["ret_off"]).abs().max()
        cmp = {
            "dry_total_ret": summary.get("total_ret"),
            "offline_total_ret": off_sum.get("total_ret"),
            "dry_n": int(len(dry)),
            "offline_n": int(len(ot)),
            "matched": int(len(both)),
            "only_dry": int((m["_merge"] == "left_only").sum()),
            "only_offline": int((m["_merge"] == "right_only").sum()),
            "max_abs_ret_diff": float(delta) if delta is not None and pd.notna(delta) else None,
            "ok": bool(
                len(both) == len(dry) == len(ot)
                and delta is not None
                and float(delta) < 1e-9
            ),
        }
        (out_dir / "compare_summary.json").write_text(json.dumps(cmp, indent=2), encoding="utf-8")
        print(json.dumps(cmp, indent=2))


if __name__ == "__main__":
    main()
