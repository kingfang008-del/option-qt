#!/usr/bin/env python3
"""Formal acceptance: freeze baseline vs Watchdog (Degrade+Halt, hunter off).

Writes scoreboard + bad-day stats under ``--out``. Does not mutate freeze.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay


def _run(prof: dict, *, start: str, end: str, tag: str, out: Path) -> dict:
    p = copy.deepcopy(prof)
    p["date_range"] = {"start": start, "end": end}
    # force hunter off for acceptance
    wd = dict(p.get("watchdog") or {})
    if wd:
        hunt = dict(wd.get("hunter") or {})
        hunt["enabled"] = False
        wd["hunter"] = hunt
        p["watchdog"] = wd
    res = run_offline_replay(p, scheme="single")
    s = res["summary"]
    daily = res["daily"].copy()
    trades = res["trades"].copy()
    sub = out / tag
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "summary.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
    daily.to_csv(sub / "daily.csv", index=False)
    trades.to_csv(sub / "trades.csv", index=False)
    return {"summary": s, "daily": daily, "trades": trades, "tag": tag}


def _bad_stats(daily: pd.DataFrame, *, thr: float = -0.03) -> dict:
    if daily is None or daily.empty:
        return {"n_bad": 0, "mean_bad": None, "sum_bad": 0.0, "n_neg": 0, "sum_neg": 0.0}
    d = daily.copy()
    d["day_ret"] = d["day_ret"].astype(float)
    bad = d[d["day_ret"] <= thr]
    neg = d[d["day_ret"] < 0]
    return {
        "n_bad": int(len(bad)),
        "mean_bad": float(bad["day_ret"].mean()) if len(bad) else None,
        "sum_bad": float(bad["day_ret"].sum()) if len(bad) else 0.0,
        "n_neg": int(len(neg)),
        "sum_neg": float(neg["day_ret"].sum()) if len(neg) else 0.0,
        "worst_day": str(d.loc[d["day_ret"].idxmin(), "date"]) if len(d) else None,
        "worst_ret": float(d["day_ret"].min()) if len(d) else None,
    }


def _day_ret(daily: pd.DataFrame, date: str) -> float | None:
    hit = daily[daily["date"].astype(str) == str(date)]
    if hit.empty:
        return None
    return float(hit.iloc[0]["day_ret"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--baseline",
        default="maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json",
    )
    ap.add_argument(
        "--watchdog",
        default="maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_watchdog_v1.json",
    )
    ap.add_argument("--out", default="maga7/results/watchdog/acceptance")
    ap.add_argument("--bad-thr", type=float, default=-0.03)
    ap.add_argument("--strong-start", default="2026-05-01")
    ap.add_argument("--strong-end", default="2026-07-17")
    ap.add_argument("--weak-start", default="2026-02-01")
    ap.add_argument("--weak-end", default="2026-04-30")
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    base = load_profile(args.baseline)
    wdog = load_profile(args.watchdog)
    windows = [
        ("strong_may_jul", args.strong_start, args.strong_end),
        ("weak_feb_apr", args.weak_start, args.weak_end),
    ]

    rows = []
    details = {}
    for wname, start, end in windows:
        print(f"=== baseline {wname} ===", flush=True)
        rb = _run(base, start=start, end=end, tag=f"baseline__{wname}", out=out)
        print(f"=== watchdog {wname} ===", flush=True)
        rw = _run(wdog, start=start, end=end, tag=f"watchdog__{wname}", out=out)

        sb, sw = rb["summary"], rw["summary"]
        bad_b = _bad_stats(rb["daily"], thr=args.bad_thr)
        bad_w = _bad_stats(rw["daily"], thr=args.bad_thr)
        d0717_b = _day_ret(rb["daily"], "2026-07-17")
        d0717_w = _day_ret(rw["daily"], "2026-07-17")

        base_ret = float(sb["total_ret"])
        wd_ret = float(sw["total_ret"])
        vs = (wd_ret / base_ret) if base_ret else None
        pass_ret = bool(vs is not None and vs >= 0.95)
        pass_dd = float(sw["maxdd"]) >= float(sb["maxdd"]) - 1e-9  # less negative or equal
        # "reduce loss": bad sum less negative (greater algebraically) or fewer bad days
        pass_bad = (bad_w["sum_bad"] >= bad_b["sum_bad"] - 1e-12) or (
            bad_w["n_bad"] <= bad_b["n_bad"]
        )
        pass_0717 = True
        if d0717_b is not None and d0717_w is not None:
            pass_0717 = d0717_w >= d0717_b - 1e-12

        row = {
            "window": wname,
            "baseline_ret": base_ret,
            "watchdog_ret": wd_ret,
            "vs_baseline": vs,
            "baseline_maxdd": float(sb["maxdd"]),
            "watchdog_maxdd": float(sw["maxdd"]),
            "baseline_n_trades": int(sb["n_trades"]),
            "watchdog_n_trades": int(sw["n_trades"]),
            "n_watchdog_days": sw.get("n_watchdog_days"),
            "watchdog_state_counts": sw.get("watchdog_state_counts"),
            "router_day_counts": sw.get("router_day_counts"),
            "baseline_bad": bad_b,
            "watchdog_bad": bad_w,
            "day_ret_0717_baseline": d0717_b,
            "day_ret_0717_watchdog": d0717_w,
            "pass_ret_95": pass_ret,
            "pass_maxdd": pass_dd,
            "pass_bad_days": pass_bad,
            "pass_0717": pass_0717,
            "pass_all": bool(pass_ret and pass_dd and pass_bad and pass_0717),
        }
        rows.append(row)
        details[wname] = {
            "baseline_tag": rb["tag"],
            "watchdog_tag": rw["tag"],
        }
        print(
            f"  vs={vs:.3f} maxdd {sb['maxdd']:.3f}->{sw['maxdd']:.3f} "
            f"bad_sum {bad_b['sum_bad']:.3f}->{bad_w['sum_bad']:.3f} "
            f"0717 {d0717_b}->{d0717_w} pass={row['pass_all']}",
            flush=True,
        )

    board = pd.DataFrame(rows)
    board.to_csv(out / "scoreboard.csv", index=False)
    overall = {
        "baseline_profile": args.baseline,
        "watchdog_profile": args.watchdog,
        "hunter": "forced_off",
        "bad_thr": args.bad_thr,
        "criteria": {
            "vs_baseline_min": 0.95,
            "maxdd_not_worse": True,
            "bad_day_sum_or_count_improved": True,
            "day_0717_not_worse": True,
        },
        "pass_all_windows": bool(all(r["pass_all"] for r in rows)),
        "windows": rows,
        "details": details,
        "verdict": (
            "ACCEPT_RESEARCH"
            if all(r["pass_all"] for r in rows)
            else "REJECT_OR_REVIEW"
        ),
        "note": "Research acceptance only; freeze stays watchdog.enabled=false until ops sign-off.",
    }
    (out / "acceptance.json").write_text(
        json.dumps(overall, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    # markdown report
    lines = [
        "# Watchdog 验收报告",
        "",
        f"- Baseline: `{args.baseline}`",
        f"- Watchdog: `{args.watchdog}` (hunter forced off)",
        f"- Verdict: **{overall['verdict']}**",
        "",
        "| 窗 | 基线 ret | WD ret | vs | 基线 MaxDD | WD MaxDD | 坏日sum A→B | 07-17 | pass |",
        "|----|----------|--------|-----|------------|----------|-------------|-------|------|",
    ]
    for r in rows:
        bb, bw = r["baseline_bad"], r["watchdog_bad"]
        lines.append(
            f"| {r['window']} | {r['baseline_ret']:.3f} | {r['watchdog_ret']:.3f} | "
            f"{r['vs_baseline']:.3f} | {r['baseline_maxdd']:.3f} | {r['watchdog_maxdd']:.3f} | "
            f"{bb['sum_bad']:.3f}→{bw['sum_bad']:.3f} (n {bb['n_bad']}→{bw['n_bad']}) | "
            f"{r['day_ret_0717_baseline']}→{r['day_ret_0717_watchdog']} | "
            f"{'YES' if r['pass_all'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "## 触发",
            "",
        ]
    )
    for r in rows:
        lines.append(
            f"- `{r['window']}`: n_watchdog_days={r['n_watchdog_days']} "
            f"states={r['watchdog_state_counts']} experts={r['router_day_counts']}"
        )
    lines.extend(["", f"明细目录: `{out}`", ""])
    (out / "ACCEPTANCE.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(overall, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
