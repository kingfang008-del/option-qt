#!/usr/bin/env python3
"""Dual-window accept: liquidity TopK rank vs soft dvol size boost (S1 baseline).

Arms (TopK seats vs size only):
  PRE      = earliest top2 (research baseline)
  DVOL     = rank_by=dollar_vol (batch re-rank; changes seats)
  CS_DVOL  = rank_by=cs_dollar_vol (causal seat gate)
  SOFT     = earliest + trade.dvol_size_scale boost (no seat change)
  DVOL_SOFT= dollar_vol rank + soft boost

Pass (pack vs PRE): strong keep>=0.85 AND weak (ret↑ OR MaxDD↑) AND july keep>=0.95
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.provenance import code_fingerprint
from maga7.common.replay import run_offline_replay

BASE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)

SOFT_CFG = {
    "enabled": True,
    "mode": "cs_rank",
    "scales": {"1": 1.25, "2": 1.15},
    "default_scale": 1.0,
    "min_scale": 1.0,
    "max_scale": 1.25,
}


def _apply(base: dict[str, Any], arm: str) -> dict[str, Any]:
    p = copy.deepcopy(base)
    sig = p.setdefault("signal", {})
    trade = p.setdefault("trade", {})
    trade.pop("dvol_size_scale", None)
    sig["top_k"] = 2
    if arm == "PRE":
        sig["rank_by"] = "earliest"
        return p
    if arm == "DVOL":
        sig["rank_by"] = "dollar_vol"
        return p
    if arm == "CS_DVOL":
        sig["rank_by"] = "cs_dollar_vol"
        return p
    if arm == "SOFT":
        sig["rank_by"] = "earliest"
        trade["dvol_size_scale"] = dict(SOFT_CFG)
        return p
    if arm == "DVOL_SOFT":
        sig["rank_by"] = "dollar_vol"
        trade["dvol_size_scale"] = dict(SOFT_CFG)
        return p
    raise SystemExit(f"unknown arm {arm}")


def _run(prof: dict[str, Any], *, start: str, end: str, tag: str, out: Path) -> dict[str, Any]:
    p = copy.deepcopy(prof)
    p["date_range"] = {"start": start, "end": end}
    res = run_offline_replay(p, scheme="single")
    sub = out / tag
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "summary.json").write_text(
        json.dumps(res["summary"], indent=2, default=str), encoding="utf-8"
    )
    res["daily"].to_csv(sub / "daily.csv", index=False)
    res["trades"].to_csv(sub / "trades.csv", index=False)
    s = res["summary"]
    tr = res["trades"]
    d22 = tr[tr["date"].astype(str) == "2026-07-22"] if not tr.empty else tr
    d22_syms = (
        ",".join(f"{a}/{b}" for a, b in zip(d22["symbol"], d22["dir"])) if len(d22) else ""
    )
    caught_nvda = bool(len(d22) and ((d22["symbol"] == "NVDA") & (d22["dir"] == "UP")).any())
    return {
        "tag": tag,
        "total_ret": float(s.get("total_ret") or 0.0),
        "maxdd": float(s.get("maxdd") or 0.0),
        "n_trades": int(s.get("n_trades") or 0),
        "trade_exp": s.get("trade_exp"),
        "n_dvol_size_boost": s.get("n_dvol_size_boost"),
        "rank_by": (p.get("signal") or {}).get("rank_by"),
        "d22_syms": d22_syms,
        "caught_nvda_up_d22": caught_nvda,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/liq_seat_accept_s1_apr_jul_jan_mar_v1",
    )
    ap.add_argument("--strong-start", default="2026-04-01")
    ap.add_argument("--strong-end", default="2026-07-22")
    ap.add_argument("--weak-start", default="2026-01-02")
    ap.add_argument("--weak-end", default="2026-03-31")
    ap.add_argument("--july-start", default="2026-07-01")
    ap.add_argument("--july-end", default="2026-07-22")
    ap.add_argument("--arms", default="PRE,DVOL,CS_DVOL,SOFT,DVOL_SOFT")
    args = ap.parse_args(argv)

    base = load_profile(BASE)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fp = code_fingerprint(base.get("_profile_path") or BASE)
    arms = [a.strip() for a in str(args.arms).split(",") if a.strip()]
    windows = [
        ("strong", args.strong_start, args.strong_end),
        ("weak", args.weak_start, args.weak_end),
        ("july", args.july_start, args.july_end),
        ("d22", "2026-07-22", "2026-07-22"),
    ]
    rows: list[dict[str, Any]] = []
    for wname, start, end in windows:
        for arm in arms:
            tag = f"{wname}_{arm}"
            print(f"[run] {tag} {start}→{end}", flush=True)
            row = _run(_apply(base, arm), start=start, end=end, tag=tag, out=out)
            row["window"] = wname
            row["arm"] = arm
            rows.append(row)
            print(
                f"  ret={row['total_ret']:+.3f} mdd={row['maxdd']:+.3f} n={row['n_trades']} "
                f"boost={row.get('n_dvol_size_boost')} d22={row.get('d22_syms')!r} "
                f"nvda={row.get('caught_nvda_up_d22')}",
                flush=True,
            )

    rdf = pd.DataFrame(rows)
    rdf.to_csv(out / "scoreboard.csv", index=False)

    def cell(window: str, arm: str, col: str = "total_ret") -> float | None:
        hit = rdf[(rdf["window"] == window) & (rdf["arm"] == arm)]
        if hit.empty:
            return None
        v = hit.iloc[0][col]
        return None if pd.isna(v) else float(v)

    def decide(pack: str) -> dict[str, Any]:
        strong_pre, weak_pre, july_pre = cell("strong", "PRE"), cell("weak", "PRE"), cell("july", "PRE")
        strong_p, weak_p, july_p = cell("strong", pack), cell("weak", pack), cell("july", pack)
        strong_keep = (
            (strong_p / strong_pre)
            if strong_pre and strong_pre > 1e-9 and strong_p is not None
            else None
        )
        weak_lift = weak_p is not None and weak_pre is not None and weak_p > weak_pre
        weak_dd_ok = (
            cell("weak", pack, "maxdd") is not None
            and cell("weak", "PRE", "maxdd") is not None
            and cell("weak", pack, "maxdd") > cell("weak", "PRE", "maxdd")
        )
        july_ok = (
            july_p is not None
            and july_pre is not None
            and july_p + 1e-9 >= july_pre * 0.95
        )
        flags: list[str] = []
        flags.append("strong_retain_ok" if strong_keep is not None and strong_keep >= 0.85 else "strong_retain_fail")
        if weak_lift:
            flags.append("weak_ret_improved")
        if weak_dd_ok:
            flags.append("weak_dd_improved")
        flags.append("july_keep_ok" if july_ok else "july_keep_fail")
        d22_caught = bool(
            rdf[(rdf.window == "d22") & (rdf.arm == pack)]["caught_nvda_up_d22"].any()
        )
        if d22_caught:
            flags.append("d22_nvda_caught")
        if (
            "strong_retain_ok" in flags
            and ("weak_ret_improved" in flags or "weak_dd_improved" in flags)
            and "july_keep_ok" in flags
        ):
            decision = "PROMOTE_LIQ_RESEARCH"
        elif "strong_retain_ok" in flags and "july_keep_ok" in flags:
            decision = "OVERLAY_ONLY"
        else:
            decision = "REJECT_FOR_BASELINE"
        return {
            "pack": pack,
            "decision": decision,
            "flags": flags,
            "metrics": {
                "strong_pre": strong_pre,
                "strong_pack": strong_p,
                "strong_keep": strong_keep,
                "weak_pre": weak_pre,
                "weak_pack": weak_p,
                "weak_maxdd_pre": cell("weak", "PRE", "maxdd"),
                "weak_maxdd_pack": cell("weak", pack, "maxdd"),
                "july_pre": july_pre,
                "july_pack": july_p,
                "d22_pre_ret": cell("d22", "PRE"),
                "d22_pack_ret": cell("d22", pack),
                "d22_nvda_caught": d22_caught,
            },
        }

    packs = [a for a in arms if a != "PRE"]
    verdicts = [decide(a) for a in packs]
    report = {
        "profile": BASE,
        "code_fingerprint": fp,
        "soft_cfg": SOFT_CFG,
        "scoreboard": rows,
        "verdicts": verdicts,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("\n=== verdicts ===", flush=True)
    for v in verdicts:
        m = v["metrics"]
        print(
            f"{v['pack']}: {v['decision']} flags={v['flags']}\n"
            f"  strong keep={m['strong_keep']}  {m['strong_pre']:+.3f}→{m['strong_pack']:+.3f}\n"
            f"  weak   {m['weak_pre']:+.3f}→{m['weak_pack']:+.3f} "
            f"mdd {m['weak_maxdd_pre']}→{m['weak_maxdd_pack']}\n"
            f"  july   {m['july_pre']:+.3f}→{m['july_pack']:+.3f}\n"
            f"  d22    {m['d22_pre_ret']:+.3f}→{m['d22_pack_ret']:+.3f} nvda={m['d22_nvda_caught']}",
            flush=True,
        )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
