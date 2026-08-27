#!/usr/bin/env python3
"""Dual-window accept for PM fade sleeve (scan stock → option fill 1DTE+).

Pass: strong total_ret>0 AND weak total_ret≥0 AND strong maxdd > −0.20
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/mnt/s990/data/maga7/results/pm_fade_accept_v1")
    ap.add_argument("--strong-start", default="2026-04-01")
    ap.add_argument("--strong-end", default="2026-07-22")
    ap.add_argument("--weak-start", default="2026-01-02")
    ap.add_argument("--weak-end", default="2026-03-31")
    ap.add_argument("--ext-mins", default="0.008,0.012")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args(argv)
    py = args.python
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prof = load_profile(
        "maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
    )
    rdir = Path(prof["_paths"]["results_dir"])

    windows = {
        "strong": (args.strong_start, args.strong_end),
        "weak": (args.weak_start, args.weak_end),
    }
    rows: list[dict[str, Any]] = []
    for wname, (start, end) in windows.items():
        ev_tag = f"research_pm_fade_{wname}"
        ev_dir = rdir / ev_tag
        if not (ev_dir / "events.parquet").is_file() and not (ev_dir / "events.csv").is_file():
            _run(
                [
                    py,
                    "-m",
                    "maga7.tools.scan_pm_fade",
                    "--start-date",
                    start,
                    "--end-date",
                    end,
                    "--tag",
                    ev_tag,
                    "--ext-mins",
                    args.ext_mins,
                    "--confirm-minutes",
                    "5",
                    "--require-confirm",
                    "--hold-minutes",
                    "15",
                ]
            )
        for ext in [float(x) for x in args.ext_mins.split(",") if x.strip()]:
            fill_tag = f"research_pm_fade_fill_{wname}_e{int(ext*10000)}"
            fill_dir = rdir / fill_tag
            if not (fill_dir / "summary.json").is_file():
                _run(
                    [
                        py,
                        "-m",
                        "maga7.tools.run_pm_fade_option_fill",
                        "--events-tag",
                        ev_tag,
                        "--tag",
                        fill_tag,
                        "--ext-min",
                        str(ext),
                        "--prefer-dte",
                        "1",
                        "--allowed-dte",
                        "1,2",
                        "--hold-minutes",
                        "15",
                        "--position-frac",
                        "0.10",
                    ]
                )
            s = json.loads((fill_dir / "summary.json").read_text(encoding="utf-8"))
            row = {
                "window": wname,
                "ext_min": ext,
                "total_ret": float(s.get("total_ret") or 0.0),
                "maxdd": float(s.get("maxdd") or 0.0),
                "n_trades": int(s.get("n_trades") or 0),
                "trade_win": s.get("trade_win"),
                "n_opt_fills": s.get("n_opt_fills"),
                "n_miss": s.get("n_miss"),
                "fill_tag": fill_tag,
            }
            rows.append(row)
            print(
                f"[{wname} ext={ext}] ret={row['total_ret']:+.3f} mdd={row['maxdd']:+.3f} n={row['n_trades']}",
                flush=True,
            )

    rdf = pd.DataFrame(rows)
    rdf.to_csv(out / "scoreboard.csv", index=False)
    decisions = []
    for ext in sorted(rdf["ext_min"].unique()):
        s = rdf[(rdf.window == "strong") & (rdf.ext_min == ext)]
        w = rdf[(rdf.window == "weak") & (rdf.ext_min == ext)]
        if s.empty or w.empty:
            continue
        sr, wr = float(s.iloc[0].total_ret), float(w.iloc[0].total_ret)
        sd = float(s.iloc[0].maxdd)
        flags = []
        flags.append("strong_profit" if sr > 0 else "strong_loss")
        flags.append("weak_profit" if wr > 0 else ("weak_flatish" if wr >= 0 else "weak_loss"))
        flags.append("strong_dd_ok" if sd > -0.20 else "strong_dd_fail")
        if "strong_profit" in flags and "weak_profit" in flags and "strong_dd_ok" in flags:
            decision = "PROMOTE_PM_SLEEVE"
        elif "strong_profit" in flags and "strong_dd_ok" in flags and "weak_loss" not in flags:
            decision = "OVERLAY_ONLY"
        else:
            decision = "REJECT_FOR_PM_SLEEVE"
        decisions.append(
            {
                "ext_min": float(ext),
                "decision": decision,
                "flags": flags,
                "strong_ret": sr,
                "weak_ret": wr,
                "strong_maxdd": sd,
            }
        )
    summary = {
        "profile": "maga7/CONFIG/strategy_profiles/pm_fade_ext8_c5_h15_dte1_v1.json",
        "decisions": decisions,
        "scoreboard": rows,
    }
    (out / "accept_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\n=== PM decisions ===", flush=True)
    for d in decisions:
        print(d, flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
