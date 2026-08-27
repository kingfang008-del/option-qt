#!/usr/bin/env python3
"""CORE C1 — climate map (no trading-logic change).

Slices research_baseline PnL by causal morning features (asof≤10:30) and
calendar windows. Answers: which climates earn the CORE edge?

Default corpus: stitch S1 research_baseline accept trades (Jan–Mar + Apr–Jul)
joined to ``regime_router/router_dataset_v2.parquet``. Optional ``--replay``
re-runs current profile offline (slower).

Does NOT promote regime actions — map only (C1). Next is C2 soft prior.

Example:
  PYTHONPATH=. python -m maga7.tools.run_core_climate_map \\
    --tag research_core_climate_map_v1
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

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
DEFAULT_ROUTER = ROOT / "maga7/results/regime_router/router_dataset_v2.parquet"
S1_ROOT = Path(
    "/mnt/s990/data/maga7/results/s1_research_baseline_accept_apr_jul_jan_mar_v1"
)
S1_ARMS = ("weak_S1", "strong_S1")


def _calendar_bucket(date: str) -> str:
    d = str(date)
    if d < "2026-04-01":
        return "jan_mar"
    if d < "2026-06-01":
        return "apr_may"
    if d < "2026-07-10":
        return "jun_jul09"
    return "jul10_23"


def _label_climates(day: pd.DataFrame) -> pd.DataFrame:
    """Add climate columns from causal 10:30 features."""
    out = day.copy()
    q = pd.to_numeric(out.get("qqq_from_prev_1030"), errors="coerce")
    out["cli_qqq"] = np.where(
        q >= 0.003, "qqq_up", np.where(q <= -0.003, "qqq_dn", "qqq_flat")
    )
    vz = pd.to_numeric(out.get("vixy_z_1030"), errors="coerce")
    out["cli_vixy"] = np.where(
        vz >= 1.0, "vixy_high", np.where(vz <= -0.5, "vixy_low", "vixy_mid")
    )
    br = pd.to_numeric(out.get("mag7_frac_above_open"), errors="coerce")
    out["cli_breadth"] = np.where(
        br >= 0.625, "breadth_up", np.where(br <= 0.375, "breadth_dn", "breadth_mid")
    )
    bdf = pd.to_numeric(out.get("breadth_dn_frac"), errors="coerce")
    out["cli_pressure"] = np.where(
        bdf >= 0.625, "press_dn", np.where(bdf <= 0.375, "press_up", "press_mid")
    )
    out["cli_calendar"] = out["date"].map(_calendar_bucket)
    return out


def _load_s1_trades() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    trades = []
    daily = []
    for arm in S1_ARMS:
        tp = S1_ROOT / arm / "trades.csv"
        dp = S1_ROOT / arm / "daily.csv"
        if not tp.exists():
            raise SystemExit(f"missing {tp}")
        t = pd.read_csv(tp)
        t["corpus_arm"] = arm
        trades.append(t)
        if dp.exists():
            d = pd.read_csv(dp)
            d["corpus_arm"] = arm
            daily.append(d)
    tr = pd.concat(trades, ignore_index=True)
    dy = pd.concat(daily, ignore_index=True) if daily else pd.DataFrame()
    note = (
        f"stitched {list(S1_ARMS)} from {S1_ROOT.name} "
        "(S1 research_baseline accept vintage; map-only)"
    )
    return tr, dy, note


def _replay_trades(profile: dict[str, Any], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    p = dict(profile)
    p["date_range"] = {"start": start, "end": end}
    print(f"offline replay {start}..{end} …", flush=True)
    res = run_offline_replay(p, scheme="single")
    return (
        res["trades"].copy(),
        res["daily"].copy(),
        f"fresh replay profile={p.get('profile_id')} {start}..{end}",
    )


def _day_pnl_from_trades(tr: pd.DataFrame) -> pd.DataFrame:
    """Approximate day portfolio contribution: sum(ret * size_frac)."""
    if tr.empty:
        return pd.DataFrame(columns=["date", "day_contrib", "n_trades", "mean_ret", "win"])
    x = tr.copy()
    x["size_frac"] = pd.to_numeric(x.get("size_frac"), errors="coerce").fillna(0.2)
    x["ret"] = pd.to_numeric(x["ret"], errors="coerce")
    x["contrib"] = x["ret"] * x["size_frac"]
    g = x.groupby("date", as_index=False).agg(
        day_contrib=("contrib", "sum"),
        n_trades=("ret", "size"),
        mean_ret=("ret", "mean"),
        win=("ret", lambda s: float((s > 0).mean())),
    )
    return g


def _slice_stats(
    trades: pd.DataFrame,
    day_feat: pd.DataFrame,
    *,
    climate_col: str,
) -> pd.DataFrame:
    t = trades.merge(day_feat[["date", climate_col]], on="date", how="left")
    t["size_frac"] = pd.to_numeric(t.get("size_frac"), errors="coerce").fillna(0.2)
    t["ret"] = pd.to_numeric(t["ret"], errors="coerce")
    t["contrib"] = t["ret"] * t["size_frac"]
    rows = []
    total_contrib = float(t["contrib"].sum()) if len(t) else 0.0
    for key, g in t.groupby(climate_col, dropna=False):
        rr = g["ret"].to_numpy(dtype=float)
        contrib = float(g["contrib"].sum())
        rows.append(
            {
                "axis": climate_col,
                "climate": str(key),
                "n_trades": int(len(g)),
                "n_days": int(g["date"].nunique()),
                "win": float((rr > 0).mean()) if len(rr) else None,
                "mean_ret": float(np.nanmean(rr)) if len(rr) else None,
                "med_ret": float(np.nanmedian(rr)) if len(rr) else None,
                "sum_contrib": contrib,
                "share_contrib": (contrib / total_contrib) if total_contrib else 0.0,
                "tpd": float(len(g) / max(g["date"].nunique(), 1)),
            }
        )
    return pd.DataFrame(rows).sort_values(["axis", "share_contrib"], ascending=[True, False])


def _wd_slice(trades: pd.DataFrame) -> pd.DataFrame:
    t = trades.copy()
    t["cli_watchdog"] = t.get("watchdog_state", "unknown").fillna("unknown").astype(str)
    t["size_frac"] = pd.to_numeric(t.get("size_frac"), errors="coerce").fillna(0.2)
    t["ret"] = pd.to_numeric(t["ret"], errors="coerce")
    t["contrib"] = t["ret"] * t["size_frac"]
    total = float(t["contrib"].sum()) if len(t) else 0.0
    rows = []
    for key, g in t.groupby("cli_watchdog"):
        rr = g["ret"].to_numpy(dtype=float)
        contrib = float(g["contrib"].sum())
        rows.append(
            {
                "axis": "cli_watchdog",
                "climate": str(key),
                "n_trades": int(len(g)),
                "n_days": int(g["date"].nunique()),
                "win": float((rr > 0).mean()) if len(rr) else None,
                "mean_ret": float(np.nanmean(rr)) if len(rr) else None,
                "med_ret": float(np.nanmedian(rr)) if len(rr) else None,
                "sum_contrib": contrib,
                "share_contrib": (contrib / total) if total else 0.0,
                "tpd": float(len(g) / max(g["date"].nunique(), 1)),
            }
        )
    return pd.DataFrame(rows).sort_values("share_contrib", ascending=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_core_climate_map_v1")
    ap.add_argument("--router-dataset", default=str(DEFAULT_ROUTER))
    ap.add_argument("--replay", action="store_true", help="Fresh offline replay (slow)")
    ap.add_argument("--start-date", default="2026-01-02")
    ap.add_argument("--end-date", default="2026-07-23")
    args = ap.parse_args(argv)

    profile = load_profile(args.profile)
    out = Path(profile["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    if args.replay:
        trades, daily, corpus_note = _replay_trades(
            profile, args.start_date, args.end_date
        )
    else:
        trades, daily, corpus_note = _load_s1_trades()

    trades["date"] = trades["date"].astype(str)
    router_path = Path(args.router_dataset)
    if not router_path.exists():
        raise SystemExit(f"missing router dataset {router_path}")
    router = pd.read_parquet(router_path)
    router["date"] = router["date"].astype(str)
    day_feat = _label_climates(router)
    # keep only dates in trade corpus range
    d0, d1 = str(trades["date"].min()), str(trades["date"].max())
    day_feat = day_feat[(day_feat["date"] >= d0) & (day_feat["date"] <= d1)].copy()

    axes = ["cli_calendar", "cli_qqq", "cli_vixy", "cli_breadth", "cli_pressure"]
    boards = [_slice_stats(trades, day_feat, climate_col=a) for a in axes]
    boards.append(_wd_slice(trades))
    scoreboard = pd.concat(boards, ignore_index=True)
    scoreboard.to_csv(out / "scoreboard.csv", index=False)

    trades_j = trades.merge(
        day_feat[
            [
                "date",
                "cli_calendar",
                "cli_qqq",
                "cli_vixy",
                "cli_breadth",
                "cli_pressure",
                "qqq_from_prev_1030",
                "vixy_z_1030",
                "mag7_frac_above_open",
                "breadth_dn_frac",
            ]
        ],
        on="date",
        how="left",
    )
    trades_j.to_csv(out / "trades_labeled.csv", index=False)
    day_pnl = _day_pnl_from_trades(trades)
    day_pnl = day_pnl.merge(
        day_feat[["date", "cli_calendar", "cli_qqq", "cli_vixy", "cli_breadth"]],
        on="date",
        how="left",
    )
    day_pnl.to_csv(out / "day_contrib.csv", index=False)

    # Headline: where does contribution come from?
    cal = scoreboard[scoreboard.axis == "cli_calendar"].copy()
    qqq = scoreboard[scoreboard.axis == "cli_qqq"].copy()
    top_share = scoreboard.sort_values("share_contrib", ascending=False).head(8)

    summary: dict[str, Any] = {
        "protocol": "core_c1_climate_map",
        "promotion": False,
        "corpus": corpus_note,
        "profile": profile.get("profile_id"),
        "date_span": {"start": d0, "end": d1},
        "n_trades": int(len(trades)),
        "n_days_traded": int(trades["date"].nunique()),
        "router_dataset": str(router_path),
        "total_sum_contrib": float(
            (pd.to_numeric(trades["ret"], errors="coerce")
             * pd.to_numeric(trades.get("size_frac"), errors="coerce").fillna(0.2)).sum()
        ),
        "calendar": cal.to_dict(orient="records"),
        "qqq_axis": qqq.to_dict(orient="records"),
        "top_share_rows": top_share.to_dict(orient="records"),
        "next_step": "C2_regime_prior_soft_scale_on_weak_climates",
        "note": (
            "sum_contrib = sum(ret*size_frac), not equity compound. "
            "Map-only — do not wire actions from this tag."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# CORE C1 — Climate map（不改交易逻辑）",
        "",
        f"- corpus: {corpus_note}",
        f"- span: {d0} .. {d1}",
        f"- n_trades: {len(trades)} · traded_days: {trades['date'].nunique()}",
        f"- metric: sum_contrib = Σ(ret×size_frac)（贡献代理，非权益复利）",
        "- promote: **NONE**（地图 only）",
        "",
        "## Calendar",
        "",
    ]
    try:
        lines.append(cal.to_markdown(index=False))
    except Exception:
        lines.append(cal.to_string(index=False))
    lines += ["", "## QQQ asof 10:30 (|fp|≥30bp)", ""]
    try:
        lines.append(qqq.to_markdown(index=False))
    except Exception:
        lines.append(qqq.to_string(index=False))
    lines += ["", "## Full scoreboard", ""]
    try:
        lines.append(scoreboard.to_markdown(index=False))
    except Exception:
        lines.append(scoreboard.to_string(index=False))
    lines += [
        "",
        "## 读法（给 C2）",
        "",
        "1. 贡献集中在哪些气候？那些是「家门口」。",
        "2. 哪类气候 n 不小但 mean/share 差？→ C2 软缩仓候选，**不是**再加硬 BLOCK。",
        "3. Watchdog 态若几乎全是 normal，说明离散适应几乎没触发。",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== CALENDAR ===", flush=True)
    print(cal.to_string(index=False), flush=True)
    print("\n=== QQQ ===", flush=True)
    print(qqq.to_string(index=False), flush=True)
    print("\n=== TOP SHARE ===", flush=True)
    print(top_share.to_string(index=False), flush=True)
    print(json.dumps({"n_trades": len(trades), "next": summary["next_step"]}, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
