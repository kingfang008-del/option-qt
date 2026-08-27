#!/usr/bin/env python3
"""CORE C2 — climate prior bakeoff (soft size scale, no Rule-A change).

Post-hoc rescale of S1 research_baseline trades using causal 10:30 features.
Same fires; only size_frac changes. Calendar is an eval window, not a live gate.

PASS vs S1 baseline (same trades):
  strong keep >= 0.95
  weak keep >= 0.95
  weak MaxDD improved (less negative)

Example:
  PYTHONPATH=. python -m maga7.tools.run_core_c2_climate_prior \\
    --tag research_core_c2_climate_prior
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.climate_prior import (
    load_climate_day_table,
    parse_climate_prior,
    resolve_climate_prior,
)
from maga7.common.config import load_profile

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
S1_ROOT = Path(
    "/mnt/s990/data/maga7/results/s1_research_baseline_accept_apr_jul_jan_mar_v1"
)
WINDOWS = (
    ("weak", "2026-01-02", "2026-03-31"),
    ("strong", "2026-04-01", "2026-07-21"),
    ("july", "2026-07-01", "2026-07-21"),
)
VARIANTS: list[dict[str, Any]] = [
    {
        "name": "vixy_high_s05",
        "cfg": {
            "enabled": True,
            "scale": 0.5,
            "combine": "or",
            "use_vixy": True,
            "use_breadth_mid": False,
        },
    },
    {
        "name": "breadth_mid_s05",
        "cfg": {
            "enabled": True,
            "scale": 0.5,
            "combine": "or",
            "use_vixy": False,
            "use_breadth_mid": True,
        },
    },
    {
        "name": "or_s05",
        "cfg": {
            "enabled": True,
            "scale": 0.5,
            "combine": "or",
            "use_vixy": True,
            "use_breadth_mid": True,
        },
    },
    {
        "name": "or_s07",
        "cfg": {
            "enabled": True,
            "scale": 0.7,
            "combine": "or",
            "use_vixy": True,
            "use_breadth_mid": True,
        },
    },
    {
        "name": "and_s05",
        "cfg": {
            "enabled": True,
            "scale": 0.5,
            "combine": "and",
            "use_vixy": True,
            "use_breadth_mid": True,
        },
    },
]


def _load_s1_trades() -> pd.DataFrame:
    parts = []
    for arm in ("weak_S1", "strong_S1"):
        p = S1_ROOT / arm / "trades.csv"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        t = pd.read_csv(p)
        t["corpus_arm"] = arm
        parts.append(t)
    out = pd.concat(parts, ignore_index=True)
    out["date"] = out["date"].astype(str)
    out["ret"] = pd.to_numeric(out["ret"], errors="coerce")
    out["size_frac"] = pd.to_numeric(out.get("size_frac"), errors="coerce").fillna(0.2)
    out["entry_ts"] = pd.to_datetime(out["entry_ts"], utc=True, errors="coerce")
    return out.sort_values(["date", "entry_ts", "symbol"]).reset_index(drop=True)


def _equity(trades: pd.DataFrame) -> dict[str, Any]:
    if trades is None or trades.empty:
        return {"n": 0, "total_ret": 0.0, "maxdd": 0.0, "win": None, "mean_ret": None, "n_scaled": 0}
    eq = 100.0
    peak = 100.0
    maxdd = 0.0
    n_scaled = 0
    rets = []
    for row in trades.itertuples(index=False):
        sf = float(row.size_frac)
        r = float(row.ret)
        if not np.isfinite(r):
            continue
        eq *= 1.0 + sf * r
        peak = max(peak, eq)
        if peak > 0:
            maxdd = min(maxdd, eq / peak - 1.0)
        rets.append(r)
        extra = getattr(row, "climate_scale", 1.0)
        if extra is not None and abs(float(extra) - 1.0) > 1e-12:
            n_scaled += 1
    rr = np.array(rets, dtype=float)
    return {
        "n": int(len(rr)),
        "total_ret": float(eq / 100.0 - 1.0),
        "maxdd": float(maxdd),
        "win": float((rr > 0).mean()) if len(rr) else None,
        "mean_ret": float(rr.mean()) if len(rr) else None,
        "n_scaled": int(n_scaled),
    }


def _apply(trades: pd.DataFrame, cfg_raw: dict[str, Any], day_table: pd.DataFrame) -> pd.DataFrame:
    cfg = parse_climate_prior(cfg_raw)
    rows = []
    for row in trades.itertuples(index=False):
        rec = row._asdict() if hasattr(row, "_asdict") else dict(zip(trades.columns, row))
        scale, reason = resolve_climate_prior(cfg, date=str(rec["date"]), day_table=day_table)
        rec["climate_scale"] = float(scale)
        rec["climate_reason"] = reason
        rec["size_frac"] = float(rec["size_frac"]) * float(scale)
        rows.append(rec)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_core_c2_climate_prior")
    ap.add_argument("--dataset", default="maga7/results/regime_router/router_dataset_v2.parquet")
    args = ap.parse_args(argv)

    profile = load_profile(args.profile)
    out = Path(profile["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    trades = _load_s1_trades()
    day_table = load_climate_day_table(args.dataset)
    if day_table is None or day_table.empty:
        raise SystemExit(f"missing climate day table {args.dataset}")

    rows = []
    baseline_by_win: dict[str, dict[str, Any]] = {}
    for wname, w0, w1 in WINDOWS:
        sub = trades[(trades["date"] >= w0) & (trades["date"] <= w1)].copy()
        st = _equity(sub)
        baseline_by_win[wname] = st
        rows.append({"variant": "baseline_s1", "window": wname, **st, "keep": 1.0})

    for var in VARIANTS:
        scaled = _apply(trades, var["cfg"], day_table)
        for wname, w0, w1 in WINDOWS:
            sub = scaled[(scaled["date"] >= w0) & (scaled["date"] <= w1)].copy()
            st = _equity(sub)
            base = baseline_by_win[wname]
            keep = (
                (1.0 + st["total_ret"]) / (1.0 + base["total_ret"])
                if (1.0 + base["total_ret"]) != 0
                else 0.0
            )
            dd_delta = float(st["maxdd"] - base["maxdd"])  # >0 means shallower DD
            rows.append(
                {
                    "variant": var["name"],
                    "window": wname,
                    **st,
                    "keep": float(keep),
                    "maxdd_delta": dd_delta,
                }
            )

    sb = pd.DataFrame(rows)
    sb.to_csv(out / "scoreboard.csv", index=False)

    def _row(variant: str, window: str) -> pd.Series | None:
        hit = sb[(sb.variant == variant) & (sb.window == window)]
        return hit.iloc[0] if len(hit) else None

    adopted = []
    for var in VARIANTS:
        name = var["name"]
        strong = _row(name, "strong")
        weak = _row(name, "weak")
        july = _row(name, "july")
        if strong is None or weak is None:
            continue
        ok = (
            float(strong["keep"]) >= 0.95
            and float(weak["keep"]) >= 0.95
            and float(weak["maxdd"] - baseline_by_win["weak"]["maxdd"]) >= 0.005
            and int(weak["n_scaled"]) >= 3
        )
        adopted.append(
            {
                "variant": name,
                "pass": bool(ok),
                "strong_keep": float(strong["keep"]),
                "weak_keep": float(weak["keep"]),
                "july_keep": float(july["keep"]) if july is not None else None,
                "weak_maxdd": float(weak["maxdd"]),
                "weak_maxdd_delta": float(weak["maxdd"] - baseline_by_win["weak"]["maxdd"]),
                "strong_total_ret": float(strong["total_ret"]),
                "weak_total_ret": float(weak["total_ret"]),
                "n_scaled_strong": int(strong["n_scaled"]),
                "n_scaled_weak": int(weak["n_scaled"]),
            }
        )
    passed = [a for a in adopted if a["pass"]]
    passed.sort(key=lambda x: (x["weak_maxdd_delta"], x["strong_keep"]), reverse=True)
    promote = f"C2_{passed[0]['variant']}" if passed else "NONE"

    summary = {
        "protocol": "core_c2_climate_prior",
        "promotion_mark": "soft_size_scale_only",
        "corpus": "s1_research_baseline_accept_apr_jul_jan_mar_v1",
        "pass_rule": "strong keep>=0.95 AND weak keep>=0.95 AND weak MaxDD Δ>=50bp AND n_scaled_weak>=3",
        "baseline": {k: v for k, v in baseline_by_win.items()},
        "variants": adopted,
        "promote": promote,
        "pass": bool(promote != "NONE"),
        "next_step": (
            "wire_climate_prior_on_research_baseline"
            if promote != "NONE"
            else "keep_baseline_try_other_causal_soft_scale"
        ),
        "note": "Calendar is eval-only. Live gate = vixy_z / mag7_frac_above_open asof 10:30.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# CORE C2 — Climate prior bakeoff",
        "",
        "- action: **soft size scale only** (no Rule-A / no BLOCK)",
        "- features: VIXY z @10:30 · Mag7 breadth mid @10:30",
        f"- promote: **{promote}**",
        f"- pass: **{summary['pass']}**",
        "",
        "## Scoreboard",
        "",
    ]
    try:
        lines.append(sb.to_markdown(index=False))
    except Exception:
        lines.append(sb.to_string(index=False))
    if promote != "NONE":
        best = passed[0]
        lines += [
            "",
            "## 结论",
            "",
            f"**C2 PASS** → `{best['variant']}` "
            f"strong keep={best['strong_keep']:.3f} · weak keep={best['weak_keep']:.3f} · "
            f"weak MaxDD Δ={best['weak_maxdd_delta']:+.3f}。",
            "可写入 research_baseline `trade.climate_prior`（默认 enabled）；生产 freeze 另闸。",
        ]
    else:
        lines += [
            "",
            "## 结论",
            "",
            "**C2 FAIL** — 软缩仓未能同时保住强窗并改善弱窗 MaxDD。不接线；不改 L0。",
        ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(sb.to_string(index=False), flush=True)
    print(json.dumps({"promote": promote, "pass": summary["pass"]}, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
