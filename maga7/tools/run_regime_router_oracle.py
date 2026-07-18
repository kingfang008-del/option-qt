#!/usr/bin/env python3
"""Oracle regime-router scoreboard: perfect day_type → expert overlays.

Upper bound before training a causal Router. Default experts = soft scale
(``experts_v1.json``); pass ``--experts ...aggressive.json`` for hard block ceiling.
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
    res = run_offline_replay(p, scheme="single")
    s = res["summary"]
    daily = res["daily"]
    trades = res["trades"]
    sub = out / tag
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "summary.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
    daily.to_csv(sub / "daily.csv", index=False)
    trades.to_csv(sub / "trades.csv", index=False)
    d0717 = None
    hit = daily[daily["date"].astype(str) == "2026-07-17"] if not daily.empty else pd.DataFrame()
    if len(hit):
        d0717 = float(hit.iloc[0]["day_ret"])
    return {
        "tag": tag,
        "total_ret": float(s["total_ret"]),
        "maxdd": float(s["maxdd"]),
        "n_trades": int(s["n_trades"]),
        "trade_win": s.get("trade_win"),
        "n_router_expert_days": s.get("n_router_expert_days"),
        "router_day_counts": s.get("router_day_counts"),
        "day_ret_0717": d0717,
        "daily": daily,
    }


def _cluster_stats(daily: pd.DataFrame, labels: dict[str, str], types: list[str]) -> dict:
    if daily is None or daily.empty:
        return {}
    d = daily.copy()
    d["date"] = d["date"].astype(str)
    d["day_type"] = d["date"].map(labels)
    out = {}
    for t in types:
        sub = d[d["day_type"] == t]
        if sub.empty:
            out[t] = {"n": 0, "mean_day_ret": None, "sum_day_ret": None}
        else:
            out[t] = {
                "n": int(len(sub)),
                "mean_day_ret": float(sub["day_ret"].mean()),
                "sum_day_ret": float(sub["day_ret"].sum()),
            }
    bad = d[d["day_type"].isin(types)]
    out["_all_labeled_bad"] = {
        "n": int(len(bad)),
        "mean_day_ret": float(bad["day_ret"].mean()) if len(bad) else None,
        "sum_day_ret": float(bad["day_ret"].sum()) if len(bad) else None,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default="maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json",
    )
    ap.add_argument("--labels", default="maga7/results/regime_router/day_type_labels.csv")
    ap.add_argument("--experts", default="maga7/CONFIG/regime_router/experts_v1.json")
    ap.add_argument("--out", default="maga7/results/regime_router/oracle_scoreboard")
    ap.add_argument("--strong-start", default="2026-05-01")
    ap.add_argument("--strong-end", default="2026-07-17")
    ap.add_argument("--weak-start", default="2026-02-01")
    ap.add_argument("--weak-end", default="2026-04-30")
    ap.add_argument(
        "--also-full-year",
        action="store_true",
        help="also run 2025-07-01 → strong-end",
    )
    args = ap.parse_args()

    # ensure labels exist
    lab_path = Path(args.labels)
    if not lab_path.is_file():
        from maga7.tools.build_regime_router_labels import main as build_labels

        sys.argv = ["build_regime_router_labels", "--out", str(lab_path)]
        build_labels()

    labels_df = pd.read_csv(lab_path)
    labels = {str(r.date): str(r.day_type) for r in labels_df.itertuples(index=False)}
    expert_types = ["rebound_trap_dn", "dn_toxic", "up_toxic"]

    base = load_profile(args.profile)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    windows = [
        ("strong_may_jul", args.strong_start, args.strong_end),
        ("weak_feb_apr", args.weak_start, args.weak_end),
    ]
    if args.also_full_year:
        windows.append(("full_year", "2025-07-01", args.strong_end))

    board = []
    for wname, start, end in windows:
        # baseline
        row_b = _run(base, start=start, end=end, tag=f"{wname}__baseline", out=out)
        base_ret = row_b["total_ret"]
        cl_b = _cluster_stats(row_b.pop("daily"), labels, expert_types)
        board.append(
            {
                **{k: v for k, v in row_b.items() if k != "daily"},
                "window": wname,
                "variant": "baseline",
                "vs_baseline": 1.0,
                "cluster": cl_b,
            }
        )

        # oracle soft
        p = copy.deepcopy(base)
        p["regime_router"] = {
            "enabled": True,
            "mode": "oracle",
            "labels_path": str(lab_path),
            "experts_path": str(args.experts),
        }
        row_o = _run(p, start=start, end=end, tag=f"{wname}__oracle", out=out)
        cl_o = _cluster_stats(row_o.pop("daily"), labels, expert_types)
        board.append(
            {
                **{k: v for k, v in row_o.items() if k != "daily"},
                "window": wname,
                "variant": "oracle",
                "vs_baseline": row_o["total_ret"] / base_ret if base_ret else None,
                "cluster": cl_o,
            }
        )

    (out / "scoreboard.json").write_text(json.dumps(board, indent=2, default=str), encoding="utf-8")
    flat = []
    for r in board:
        flat.append(
            {
                "window": r["window"],
                "variant": r["variant"],
                "total_ret": r["total_ret"],
                "maxdd": r["maxdd"],
                "n_trades": r["n_trades"],
                "vs_baseline": r["vs_baseline"],
                "day_ret_0717": r.get("day_ret_0717"),
                "n_router_expert_days": r.get("n_router_expert_days"),
                "cluster_mean_rebound": (r.get("cluster") or {}).get("rebound_trap_dn", {}).get("mean_day_ret"),
                "cluster_mean_dn_toxic": (r.get("cluster") or {}).get("dn_toxic", {}).get("mean_day_ret"),
                "cluster_mean_up_toxic": (r.get("cluster") or {}).get("up_toxic", {}).get("mean_day_ret"),
                "cluster_mean_all_bad": (r.get("cluster") or {}).get("_all_labeled_bad", {}).get("mean_day_ret"),
            }
        )
    pd.DataFrame(flat).to_csv(out / "scoreboard.csv", index=False)
    print(pd.DataFrame(flat).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
