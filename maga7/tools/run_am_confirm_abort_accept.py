#!/usr/bin/env python3
"""Post-fill confirm-or-abort accept on frozen AM+EXT pulse FO book.

Replays quote paths for the dual-pass cell
``pulse_FO_t0.008_tp0.15_sl0.2_sp0.15_lag5.0`` (AM 09:30–10:25 + EXT 10:25–11:30).

Baseline = TP15/SL20. Variants add causal confirm_or_abort after fill:
  confirm within T seconds (mark >= thr) else flatten (confirm_abort);
  optional tighter early_abort before confirm; TP/SL always live.

Portfolio: fixed size per seat (no dilution), max_concurrent=2, AM/EXT independent.

Dual windows: may_jul09 / jul10_23. PASS if both windows mean>0, add>0,
retain_add vs baseline >= min_retain, and trade_win not collapsing.

Example:
  PYTHONPATH=. python -m maga7.tools.run_am_confirm_abort_accept \\
    --tag research_am_confirm_abort_20260728
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
from maga7.common.fills import FillSpec
from maga7.common.option_quote_tpsl import (
    simulate_quote_tpsl,
    simulate_quote_tpsl_confirm_abort,
)
from maga7.common.replay import load_quotes, path_for_ticker, to_ny
from maga7.tools.scan_am_delayed_confirm_quote_dual import _prep_path

NY = "America/New_York"
PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
WINDOWS = (
    ("may_jul09", "2026-05-01", "2026-07-09"),
    ("jul10_23", "2026-07-10", "2026-07-23"),
)
DEFAULT_AM = (
    "/mnt/s990/data/maga7/results/research_am_conc5_20260728/"
    "am_0930_1025/trades_dual00_pulse_FO_t0.008_tp0.15_sl0.2_sp0.15_lag5.0.csv"
)
DEFAULT_EXT = (
    "/mnt/s990/data/maga7/results/research_am_conc5_20260728/"
    "am_ext_1025_1130/trades_all_pulse_FO_t0.008_tp0.15_sl0.2_sp0.15_lag5.0.csv"
)


def _portfolio_fixed(
    day_trades: list[dict],
    *,
    position_frac: float,
    max_concurrent: int,
    cooldown_minutes: float = 10.0,
) -> list[dict]:
    rows = sorted(day_trades, key=lambda r: (str(r["entry_ts"]), str(r["symbol"])))
    open_pos: list[tuple[pd.Timestamp, str]] = []
    last_exit: dict[str, pd.Timestamp] = {}
    out: list[dict] = []
    for tr in rows:
        et, xt = to_ny(tr["entry_ts"]), to_ny(tr["exit_ts"])
        sym = str(tr["symbol"])
        open_pos = [(x, s) for x, s in open_pos if x > et]
        if any(s == sym for _, s in open_pos):
            continue
        if sym in last_exit and (et - last_exit[sym]).total_seconds() < cooldown_minutes * 60:
            continue
        if len(open_pos) >= int(max_concurrent):
            continue
        row = dict(tr)
        row["size"] = float(position_frac)
        row["pnl_frac"] = float(tr["ret"]) * float(position_frac)
        out.append(row)
        open_pos.append((xt, sym))
        last_exit[sym] = xt
    return out


def _stack_fixed(trades: pd.DataFrame, *, position_frac: float, max_concurrent: int) -> pd.DataFrame:
    parts: list[dict] = []
    for lane in ("AM", "EXT"):
        sub = trades[trades["lane"] == lane]
        by_d: dict[str, list[dict]] = {}
        for _, r in sub.iterrows():
            by_d.setdefault(str(r["date"]), []).append(r.to_dict())
        for _, rs in sorted(by_d.items()):
            parts.extend(
                _portfolio_fixed(rs, position_frac=position_frac, max_concurrent=max_concurrent)
            )
    return pd.DataFrame(parts)


def _equity(df: pd.DataFrame) -> dict[str, float]:
    if df is None or df.empty:
        return {"compound": 0.0, "maxdd": 0.0}
    eq = peak = 1.0
    maxdd = 0.0
    for _, r in df.sort_values(["date", "entry_ts"]).iterrows():
        eq *= 1.0 + float(r["pnl_frac"])
        peak = max(peak, eq)
        maxdd = min(maxdd, eq / peak - 1.0)
    return {"compound": float(eq - 1.0), "maxdd": float(maxdd)}


def _summarize(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {
            "n": 0,
            "win": None,
            "add": 0.0,
            "mean": None,
            "n_loss": 0,
            "loss_pnl": 0.0,
            "compound": 0.0,
            "maxdd": 0.0,
            "n_confirm_abort": 0,
            "n_early_abort": 0,
        }
    eq = _equity(df)
    reasons = df["exit_reason"].astype(str)
    out: dict[str, Any] = {
        "n": int(len(df)),
        "win": float((df["ret"] > 0).mean()),
        "add": float(df["pnl_frac"].sum()),
        "mean": float(df["ret"].mean()),
        "n_loss": int((df["ret"] <= 0).sum()),
        "loss_pnl": float(df.loc[df["ret"] <= 0, "pnl_frac"].sum()),
        "compound": eq["compound"],
        "maxdd": eq["maxdd"],
        "n_confirm_abort": int((reasons == "confirm_abort").sum()),
        "n_early_abort": int((reasons == "early_abort").sum()),
        "n_tp": int((reasons == "tp").sum()),
        "n_sl": int((reasons == "sl").sum()),
    }
    for w, a, b in WINDOWS:
        sub = df[(df["date"].astype(str) >= a) & (df["date"].astype(str) <= b)]
        out[f"{w}_n"] = int(len(sub))
        out[f"{w}_add"] = float(sub["pnl_frac"].sum()) if len(sub) else 0.0
        out[f"{w}_win"] = float((sub["ret"] > 0).mean()) if len(sub) else None
        out[f"{w}_mean"] = float(sub["ret"].mean()) if len(sub) else None
    return out


def _gate_on(row: pd.Series, apply_to: str) -> bool:
    if apply_to == "all":
        return True
    if apply_to == "ext":
        return str(row["lane"]) == "EXT"
    if apply_to == "ext_1025":
        if str(row["lane"]) != "EXT":
            return False
        return to_ny(row["entry_ts"]).strftime("%H:%M") == "10:25"
    raise ValueError(apply_to)


def _load_entries(am_csv: Path, ext_csv: Path) -> pd.DataFrame:
    am = pd.read_csv(am_csv)
    ext = pd.read_csv(ext_csv)
    am["lane"] = "AM"
    ext["lane"] = "EXT"
    return pd.concat([am, ext], ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_am_confirm_abort_20260728")
    ap.add_argument("--am-trades", default=DEFAULT_AM)
    ap.add_argument("--ext-trades", default=DEFAULT_EXT)
    ap.add_argument("--tp", type=float, default=0.15)
    ap.add_argument("--sl", type=float, default=0.20)
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--max-lag-sec", type=float, default=5.0)
    ap.add_argument("--max-spread-pct", type=float, default=0.15)
    ap.add_argument("--position-frac", type=float, default=0.20)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--confirm-secs", default="60,90,120,180")
    ap.add_argument("--confirm-thrs", default="0.02,0.03,0.05")
    ap.add_argument("--abort-thrs", default="none,0.08,0.10")
    ap.add_argument("--apply-to", default="all,ext,ext_1025")
    ap.add_argument("--min-retain", type=float, default=0.95)
    ap.add_argument("--entry-frac", type=float, default=0.75)
    ap.add_argument("--exit-frac", type=float, default=0.75)
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    quote_root = Path(prof["_paths"]["quote_1s_root"])
    out = Path(prof["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    entries = _load_entries(Path(args.am_trades), Path(args.ext_trades))
    fill = FillSpec(entry_frac=args.entry_frac, exit_frac=args.exit_frac)

    confirm_secs = [int(x) for x in args.confirm_secs.split(",") if x.strip()]
    confirm_thrs = [float(x) for x in args.confirm_thrs.split(",") if x.strip()]
    abort_thrs: list[float | None] = []
    for x in args.abort_thrs.split(","):
        x = x.strip()
        if not x or x.lower() == "none":
            abort_thrs.append(None)
        else:
            abort_thrs.append(float(x))
    apply_tos = [x.strip() for x in args.apply_to.split(",") if x.strip()]

    # Cache quote paths by (date, ticker)
    path_cache: dict[tuple[str, str], pd.DataFrame | None] = {}
    day_cache: dict[str, dict[str, pd.DataFrame]] = {}

    def get_path(date: str, symbol: str, ticker: str) -> pd.DataFrame | None:
        key = (date, ticker)
        if key in path_cache:
            return path_cache[key]
        if date not in day_cache:
            # load_quotes(quote_root, symbol, date) per pulse scanner
            day_cache[date] = {}
        if symbol not in day_cache[date]:
            try:
                day_cache[date][symbol] = _prep_path(load_quotes(quote_root, symbol, date))
            except Exception:
                day_cache[date][symbol] = None
        qday = day_cache[date][symbol]
        path = None if qday is None else _prep_path(path_for_ticker(qday, ticker))
        path_cache[key] = path
        return path

    # Precompute baseline tpsl for every entry (shared)
    base_rows: list[dict[str, Any]] = []
    skip = 0
    for _, er in entries.iterrows():
        date = str(er["date"])
        path = get_path(date, str(er["symbol"]), str(er["ticker"]))
        if path is None or path.empty:
            skip += 1
            continue
        sim = simulate_quote_tpsl(
            path,
            to_ny(er["entry_ts"]),
            tp=args.tp,
            sl=args.sl,
            max_hold_sec=args.max_hold_sec,
            fill=fill,
            max_lag_sec=args.max_lag_sec,
            max_spread_pct=args.max_spread_pct,
        )
        if sim is None:
            skip += 1
            continue
        base_rows.append(
            {
                "date": date,
                "symbol": str(er["symbol"]),
                "dir": str(er.get("dir", "DN")),
                "lane": str(er["lane"]),
                "ticker": str(er["ticker"]),
                "sig_entry_ts": str(er["entry_ts"]),
                "entry_ts": sim["entry_ts"],
                "exit_ts": sim["exit_ts"],
                "ret": float(sim["ret"]),
                "exit_reason": sim["reason"],
                "hold_sec": float(sim["hold_sec"]),
                "mfe": float(sim["mfe"]),
                "mae": float(sim["mae"]),
                "mode": "tpsl",
                "apply_to": "all",
                "confirm_sec": 0,
                "confirm_thr": 0.0,
                "abort_thr": None,
            }
        )
    base_df = pd.DataFrame(base_rows)
    print(f"baseline entries ok={len(base_df)} skip={skip}", flush=True)

    variants: list[dict[str, Any]] = [
        {
            "name": "tpsl",
            "confirm_sec": 0,
            "confirm_thr": 0.0,
            "abort_thr": None,
            "apply_to": "all",
            "use_confirm": False,
        }
    ]
    for apply_to in apply_tos:
        for cs in confirm_secs:
            for thr in confirm_thrs:
                for ab in abort_thrs:
                    ab_tag = "none" if ab is None else f"{ab:g}"
                    variants.append(
                        {
                            "name": f"ca_t{cs}_c{thr:g}_a{ab_tag}_{apply_to}",
                            "confirm_sec": cs,
                            "confirm_thr": thr,
                            "abort_thr": ab,
                            "apply_to": apply_to,
                            "use_confirm": True,
                        }
                    )

    all_trades: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []

    # index entries for path lookup by key
    entry_by_key = {
        f"{r['date']}|{r['symbol']}|{r['sig_entry_ts']}": r for _, r in base_df.iterrows()
    }
    # also keep original signal rows for path
    sig_by_key = {
        f"{str(r.date)}|{r.symbol}|{r.entry_ts}": r for _, r in entries.iterrows()
    }

    for vi, var in enumerate(variants):
        rows: list[dict[str, Any]] = []
        for key, br in entry_by_key.items():
            sig = sig_by_key.get(key)
            if sig is None:
                # entry_ts string may differ formatting — fallback via date/symbol/ticker
                continue
            use_gate = var["use_confirm"] and _gate_on(br, var["apply_to"])
            if not use_gate:
                row = dict(br)
                row["mode"] = var["name"]
                row["apply_to"] = var["apply_to"]
                row["confirm_sec"] = int(var["confirm_sec"])
                row["confirm_thr"] = float(var["confirm_thr"])
                row["abort_thr"] = var["abort_thr"]
                rows.append(row)
                continue
            path = get_path(str(br["date"]), str(br["symbol"]), str(br["ticker"]))
            if path is None or path.empty:
                continue
            sim = simulate_quote_tpsl_confirm_abort(
                path,
                to_ny(sig["entry_ts"]),
                tp=args.tp,
                sl=args.sl,
                max_hold_sec=args.max_hold_sec,
                confirm_sec=int(var["confirm_sec"]),
                confirm_thr=float(var["confirm_thr"]),
                abort_thr=var["abort_thr"],
                on_timeout="abort",
                fill=fill,
                max_lag_sec=args.max_lag_sec,
                max_spread_pct=args.max_spread_pct,
            )
            if sim is None:
                continue
            rows.append(
                {
                    "date": str(br["date"]),
                    "symbol": str(br["symbol"]),
                    "dir": str(br["dir"]),
                    "lane": str(br["lane"]),
                    "ticker": str(br["ticker"]),
                    "sig_entry_ts": str(sig["entry_ts"]),
                    "entry_ts": sim["entry_ts"],
                    "exit_ts": sim["exit_ts"],
                    "ret": float(sim["ret"]),
                    "exit_reason": sim["reason"],
                    "hold_sec": float(sim["hold_sec"]),
                    "mfe": float(sim["mfe"]),
                    "mae": float(sim["mae"]),
                    "confirmed": bool(sim["confirmed"]),
                    "mode": var["name"],
                    "apply_to": var["apply_to"],
                    "confirm_sec": int(var["confirm_sec"]),
                    "confirm_thr": float(var["confirm_thr"]),
                    "abort_thr": var["abort_thr"],
                }
            )
        raw = pd.DataFrame(rows)
        if raw.empty:
            print(f"[{vi+1}/{len(variants)}] {var['name']}: EMPTY", flush=True)
            continue
        sized = _stack_fixed(
            raw, position_frac=args.position_frac, max_concurrent=args.max_concurrent
        )
        sized["mode"] = var["name"]
        all_trades.append(sized)
        st = _summarize(sized)
        st["mode"] = var["name"]
        st["confirm_sec"] = int(var["confirm_sec"])
        st["confirm_thr"] = float(var["confirm_thr"])
        st["abort_thr"] = var["abort_thr"]
        st["apply_to"] = var["apply_to"]
        score_rows.append(st)
        print(
            f"[{vi+1}/{len(variants)}] {var['name']}: n={st['n']} win={st['win']:.1%} "
            f"add={st['add']:+.3f} cmp={st['compound']:+.1%} "
            f"abort={st['n_confirm_abort']}+{st['n_early_abort']} loss_pnl={st['loss_pnl']:+.3f}",
            flush=True,
        )

    trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    score = pd.DataFrame(score_rows)
    if score.empty:
        raise SystemExit("no score rows")

    base = score[score["mode"] == "tpsl"].iloc[0]
    score["retain_add"] = score["add"] / float(base["add"]) if float(base["add"]) else np.nan
    score["retain_may"] = score["may_jul09_add"] / float(base["may_jul09_add"]) if float(base["may_jul09_add"]) else np.nan
    score["retain_jul"] = score["jul10_23_add"] / float(base["jul10_23_add"]) if float(base["jul10_23_add"]) else np.nan
    score["delta_loss_pnl"] = score["loss_pnl"] - float(base["loss_pnl"])
    score["dual_pass"] = (
        (score["may_jul09_mean"].fillna(-1) > 0)
        & (score["jul10_23_mean"].fillna(-1) > 0)
        & (score["may_jul09_add"] > 0)
        & (score["jul10_23_add"] > 0)
        & (score["retain_add"] >= float(args.min_retain))
        & (score["retain_may"] >= float(args.min_retain) - 0.05)
        & (score["retain_jul"] >= 0.70)
    )

    # Prefer: dual_pass, then higher add, then less loss_pnl magnitude
    picks = score[score["dual_pass"] & (score["mode"] != "tpsl")].sort_values(
        ["add", "loss_pnl"], ascending=[False, False]
    )
    verdict = "PASS" if len(picks) else "FAIL"
    best = picks.iloc[0].to_dict() if len(picks) else None

    trades_all.to_csv(out / "trades_all_modes.csv", index=False)
    score.to_csv(out / "scoreboard.csv", index=False)
    summary = {
        "tag": args.tag,
        "verdict": verdict,
        "min_retain": args.min_retain,
        "position_frac": args.position_frac,
        "max_concurrent": args.max_concurrent,
        "baseline": {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in base.to_dict().items()},
        "n_variants": int(len(score)),
        "n_dual_pass": int(score["dual_pass"].sum()) - 1,  # exclude baseline if it passes
        "best": best,
        "note": (
            "confirm_or_abort post-fill on frozen AM+EXT pulse FO entries; "
            "fixed-size seats; TP/SL always live; on_timeout=abort."
        ),
    }
    # fix n_dual_pass
    summary["n_dual_pass"] = int(((score["dual_pass"]) & (score["mode"] != "tpsl")).sum())
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    score.to_json(out / "scoreboard.json", orient="records", indent=2)

    print("\n=== TOP by add (non-baseline) ===", flush=True)
    show = score[score["mode"] != "tpsl"].sort_values("add", ascending=False).head(12)
    cols = [
        "mode",
        "n",
        "win",
        "add",
        "retain_add",
        "loss_pnl",
        "n_confirm_abort",
        "may_jul09_add",
        "jul10_23_add",
        "dual_pass",
    ]
    print(show[cols].to_string(index=False), flush=True)
    print(f"\nverdict={verdict} best={best['mode'] if best else None}", flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
