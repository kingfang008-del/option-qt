#!/usr/bin/env python3
"""Classify hold-window stock paths (1s) into whipsaw subtypes.

Uses paths.stock_1s_root — no trade-tape required.

Example:
  python -m maga7.tools.analyze_stock_path_whipsaw \\
    --trades maga7/results/research_extend_mtm_full_day_peer3_l2_tt1_05_sl55_tt600d_feb_jul/trades.csv \\
    --out-dir maga7/results/research_extend_mtm_full_day_peer3_l2_tt1_05_sl55_tt600d_feb_jul/stock_path_whipsaw
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
from maga7.common.stock_path_whipsaw import analyze_hold_path

DEFAULT_PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
DEFAULT_TRADES = (
    "maga7/results/research_extend_mtm_full_day_peer3_l2_tt1_05_sl55_tt600d_feb_jul/"
    "trades.csv"
)
DEFAULT_MODES = (
    "maga7/results/research_extend_mtm_full_day_peer3_l2_tt1_05_sl55_tt600d_feb_jul/"
    "big_loss_days_modes.csv"
)


def _summary_tables(df: pd.DataFrame) -> dict:
    out: dict = {}
    if df.empty:
        return out
    g = df.groupby("subtype", dropna=False)
    out["by_subtype"] = (
        g.agg(
            n=("ret", "count"),
            mean_ret=("ret", "mean"),
            median_ret=("ret", "median"),
            win_rate=("ret", lambda s: float((s > 0).mean())),
            mean_mae_bp=("mae", lambda s: float(s.mean() * 1e4) if s.notna().any() else None),
            mean_exit_bp=("signed_exit", lambda s: float(s.mean() * 1e4) if s.notna().any() else None),
            mean_opt_ret=("ret", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    # losers / winners
    for label, mask in (
        ("losers", df["ret"] <= 0),
        ("winners", df["ret"] > 0),
        ("big_loss_trades", df.get("is_big_loss_trade", False)),
    ):
        sub = df[mask] if label != "big_loss_trades" or "is_big_loss_trade" in df.columns else df.iloc[0:0]
        if sub.empty:
            out[label] = {}
            continue
        vc = sub["subtype"].value_counts(dropna=False)
        out[label] = {
            "n": int(len(sub)),
            "subtype_counts": {str(k): int(v) for k, v in vc.items()},
            "mean_ret": float(sub["ret"].mean()),
        }
    # horizon early adverse among losers
    losers = df[df["ret"] <= 0]
    if not losers.empty:
        out["loser_early"] = {
            "frac_h1_mae_le_m15bp": float((losers["h1_mae"] <= -0.0015).mean()),
            "frac_h5_mae_le_m15bp": float((losers["h5_mae"] <= -0.0015).mean()),
            "frac_h15_recover_ok": float(
                ((losers["mae"] <= -0.0003) & (losers["h15_signed"] >= -0.0003)).mean()
            ),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, default=Path(DEFAULT_TRADES))
    ap.add_argument("--modes", type=Path, default=Path(DEFAULT_MODES))
    ap.add_argument("--profile", type=Path, default=Path(DEFAULT_PROFILE))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--stock-1s-root", type=Path, default=None)
    args = ap.parse_args()

    prof = load_profile(str(args.profile))
    stock_root = Path(args.stock_1s_root or prof["_paths"]["stock_1s_root"])
    trades = pd.read_csv(args.trades)
    trades["date"] = pd.to_datetime(trades["date"]).dt.strftime("%Y-%m-%d")
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], utc=True, format="mixed")
    trades["exit_ts"] = pd.to_datetime(trades["exit_ts"], utc=True, format="mixed")

    modes = None
    if args.modes.is_file():
        modes = pd.read_csv(args.modes)
        modes["date"] = pd.to_datetime(modes["date"]).dt.strftime("%Y-%m-%d")
        key = ["date", "symbol", "dir"]
        modes = modes.drop_duplicates(key)
        trades = trades.merge(
            modes[key + ["fail", "stk_day", "path_mfe", "path_mae"]],
            on=key,
            how="left",
            suffixes=("", "_mode"),
        )
        trades["is_big_loss_trade"] = trades["fail"].notna()
    else:
        trades["is_big_loss_trade"] = False
        trades["fail"] = None

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = args.trades.parent / "stock_path_whipsaw"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[tuple[str, str], pd.DataFrame] = {}
    rows = []
    for i, r in trades.iterrows():
        sym = str(r["symbol"]).upper()
        day = str(r["date"])
        ck = (sym, day)
        if ck not in cache:
            cache[ck] = load_stock_1s_day(stock_root, sym, day)
        m = analyze_hold_path(
            cache[ck],
            entry_ts=r["entry_ts"],
            exit_ts=r["exit_ts"],
            direction=str(r["dir"]),
        )
        d = m.as_dict()
        d.update(
            {
                "date": day,
                "symbol": sym,
                "dir": r["dir"],
                "ret": float(r["ret"]),
                "reason": r.get("reason"),
                "size_frac": r.get("size_frac"),
                "entry_ts": str(r["entry_ts"]),
                "exit_ts": str(r["exit_ts"]),
                "fail_prior": r.get("fail"),
                "is_big_loss_trade": bool(r.get("is_big_loss_trade")),
                "stk_day_prior": r.get("stk_day"),
            }
        )
        rows.append(d)
        if (len(rows) % 25) == 0:
            print(f"… {len(rows)}/{len(trades)}", flush=True)

    df = pd.DataFrame(rows)
    path_csv = out_dir / "trade_path_subtypes.csv"
    df.to_csv(path_csv, index=False)

    summary = _summary_tables(df)
    # crosstab prior fail × subtype for big-loss set
    big = df[df["is_big_loss_trade"]]
    if not big.empty and big["fail_prior"].notna().any():
        ct = pd.crosstab(big["fail_prior"], big["subtype"])
        summary["big_loss_fail_x_subtype"] = ct.to_dict()
        ct.to_csv(out_dir / "big_loss_fail_x_subtype.csv")

    # compact comparison winners vs losers subtype share
    for tag, part in (("all", df), ("losers", df[df.ret <= 0]), ("winners", df[df.ret > 0])):
        if part.empty:
            continue
        share = (part["subtype"].value_counts(normalize=True) * 100).round(1)
        summary.setdefault("subtype_share_pct", {})[tag] = share.to_dict()

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # markdown brief
    lines = [
        "# Stock 1s hold-path whipsaw typology",
        "",
        f"- trades: `{args.trades}`",
        f"- stock_1s_root: `{stock_root}`",
        f"- n_trades: {len(df)}  missing: {int((df.subtype=='missing').sum())}",
        "",
        "## Subtype share (%)",
        "",
    ]
    shares = summary.get("subtype_share_pct", {})
    if shares:
        keys = sorted({k for d in shares.values() for k in d})
        hdr = "| subtype | " + " | ".join(shares.keys()) + " |"
        sep = "|---|---" + "|---" * (len(shares) - 1) + "|"
        lines += [hdr, sep]
        for k in keys:
            cells = [f"{shares[t].get(k, 0):.1f}" for t in shares]
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "## Thresholds",
        "",
        "- shallow adverse: mae ≤ −3bp",
        "- deep adverse: mae ≤ −15bp",
        "- recover: exit ≥ −3bp OR recover ≥ 60% of mae depth",
        "",
        f"Artifacts: `{path_csv}`",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary.get("subtype_share_pct", {}), indent=2))
    print(f"wrote {path_csv}")
    print(f"wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
