#!/usr/bin/env python3
"""AM sleeve exit stress gate (pulse FO) — NOT CORE Rule-A.

Frozen AM+EXT pulse FO entries × path-state exit variants. For CORE 10:30+
use ``maga7.tools.run_exit_stress_gate`` instead.

Example:
  PYTHONPATH=. python -m maga7.tools.run_am_exit_stress_gate \\
    --tag research_am_exit_stress_gate_20260728
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
from maga7.common.option_quote_exit_stress import (
    ExitStressPolicy,
    policy_preset,
    simulate_quote_exit_stress,
)
from maga7.common.replay import load_quotes, path_for_ticker, to_ny
from maga7.tools.scan_am_delayed_confirm_quote_dual import _prep_path

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)

# Primary dual (promotion) + secondary report windows
WINDOWS = (
    ("jan_mar", "2026-01-01", "2026-03-31"),
    ("apr", "2026-04-01", "2026-04-30"),
    ("may_jul09", "2026-05-01", "2026-07-09"),
    ("jul10_23", "2026-07-10", "2026-07-23"),
)
PRIMARY_DUAL = ("may_jul09", "jul10_23")

DEFAULT_SOURCES = (
    # lane|csv
    (
        "AM",
        "/mnt/s990/data/maga7/results/research_am_pulse_oos_weak_2025h2_20260728/"
        "am_jan_mar/trades_dual00_pulse_FO_t0.008_tp0.15_sl0.2_sp0.15_lag5.0.csv",
    ),
    (
        "EXT",
        "/mnt/s990/data/maga7/results/research_am_pulse_oos_weak_2025h2_20260728/"
        "am_ext_jan_mar/trades_all_pulse_FO_t0.008_tp0.15_sl0.2_sp0.15_lag5.0.csv",
    ),
    (
        "AM",
        "/mnt/s990/data/maga7/results/research_am_apr_fill_20260728/"
        "am_0930_1025/trades_dual00_pulse_FO_t0.008_tp0.15_sl0.2_sp0.15_lag5.0.csv",
    ),
    (
        "EXT",
        "/mnt/s990/data/maga7/results/research_am_apr_fill_20260728/"
        "am_ext_1025_1130/trades_all_pulse_FO_t0.008_tp0.15_sl0.2_sp0.15_lag5.0.csv",
    ),
    (
        "AM",
        "/mnt/s990/data/maga7/results/research_am_conc5_20260728/"
        "am_0930_1025/trades_dual00_pulse_FO_t0.008_tp0.15_sl0.2_sp0.15_lag5.0.csv",
    ),
    (
        "EXT",
        "/mnt/s990/data/maga7/results/research_am_conc5_20260728/"
        "am_ext_1025_1130/trades_all_pulse_FO_t0.008_tp0.15_sl0.2_sp0.15_lag5.0.csv",
    ),
)

DEFAULT_VARIANTS = (
    "tpsl,"
    "ca_ext1025,"
    "gb08_p10,"
    "gb08_p08,"
    "gb08_green,"
    "be_lock08,"
    "trail10_8,"
    "ladder_08_03,"
    "fast_lad10_5_180,"
    "hard_sl12,"
    "ca_gb08_p10_ext1025"
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
    if trades is None or trades.empty:
        return pd.DataFrame()
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


def _window_slice(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    d = df["date"].astype(str)
    return df[(d >= a) & (d <= b)]


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
        }
    eq = _equity(df)
    reasons = df["exit_reason"].astype(str) if "exit_reason" in df.columns else pd.Series([], dtype=str)
    out: dict[str, Any] = {
        "n": int(len(df)),
        "win": float((df["ret"] > 0).mean()),
        "add": float(df["pnl_frac"].sum()),
        "mean": float(df["ret"].mean()),
        "n_loss": int((df["ret"] <= 0).sum()),
        "loss_pnl": float(df.loc[df["ret"] <= 0, "pnl_frac"].sum()),
        "compound": eq["compound"],
        "maxdd": eq["maxdd"],
        "n_tp": int((reasons == "tp").sum()) if len(reasons) else 0,
        "n_sl": int((reasons == "sl").sum()) if len(reasons) else 0,
        "n_giveback": int((reasons == "giveback").sum()) if len(reasons) else 0,
        "n_confirm_abort": int((reasons == "confirm_abort").sum()) if len(reasons) else 0,
        "n_early_abort": int((reasons == "early_abort").sum()) if len(reasons) else 0,
    }
    for w, a, b in WINDOWS:
        sub = _window_slice(df, a, b)
        out[f"{w}_n"] = int(len(sub))
        out[f"{w}_add"] = float(sub["pnl_frac"].sum()) if len(sub) else 0.0
        out[f"{w}_win"] = float((sub["ret"] > 0).mean()) if len(sub) else None
        out[f"{w}_mean"] = float(sub["ret"].mean()) if len(sub) else None
        out[f"{w}_maxdd"] = _equity(sub)["maxdd"] if len(sub) else 0.0
    return out


def _trade_key(r: pd.Series | dict) -> str:
    if isinstance(r, dict):
        return f"{r['date']}|{r['symbol']}|{r['lane']}|{r['ticker']}|{r['sig_entry_ts']}"
    return f"{r['date']}|{r['symbol']}|{r['lane']}|{r['ticker']}|{r['sig_entry_ts']}"


def _classify_playbook(mfe: float, ret: float, lane: str, entry_ts: Any) -> str:
    if mfe < 0.02:
        return "never_green"
    if mfe >= 0.08 and ret <= 0.0:
        return "gave_back"
    if str(lane) == "EXT" and to_ny(entry_ts).strftime("%H:%M") == "10:25":
        return "open_toxic"
    if ret <= 0.0:
        return "shallow_loss"
    return "winner"


def _load_entries(sources: list[tuple[str, str]]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for lane, path in sources:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_csv(p)
        df["lane"] = lane
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    # de-dupe identical signal keys across overlapping sources
    out["_k"] = (
        out["date"].astype(str)
        + "|"
        + out["symbol"].astype(str)
        + "|"
        + out["lane"].astype(str)
        + "|"
        + out["ticker"].astype(str)
        + "|"
        + out["entry_ts"].astype(str)
    )
    out = out.drop_duplicates("_k", keep="first").drop(columns=["_k"])
    return out


def _parse_variant(spec: str, *, tp: float, sl: float) -> tuple[str, ExitStressPolicy, str]:
    """Return (display_name, policy, apply_to).

    apply_to: all | ext_1025 — scopes confirm (and combo confirm) only.
    """
    name = spec.strip()
    apply_to = "all"
    if name == "ca_ext1025":
        apply_to = "ext_1025"
        core = "ca_t60_c02_a08"
    elif name.endswith("_ext1025"):
        apply_to = "ext_1025"
        core = name[: -len("_ext1025")]
    else:
        core = name

    if core in ("ca_gb08_p10", "ca_t60_c02_a08") or core.startswith("ca_"):
        if core == "ca_gb08_p10":
            pol = policy_preset("ca_gb08_p10", tp=tp, sl=sl)
        else:
            pol = policy_preset("ca_t60_c02_a08", tp=tp, sl=sl)
    else:
        pol = policy_preset(core, tp=tp, sl=sl)
    # keep display name as requested
    return name, replace_name(pol, name), apply_to


def replace_name(pol: ExitStressPolicy, name: str) -> ExitStressPolicy:
    from dataclasses import replace as dc_replace

    return dc_replace(pol, name=name)


def _apply_gate(row: pd.Series, apply_to: str) -> bool:
    if apply_to == "all":
        return True
    if apply_to == "ext_1025":
        if str(row["lane"]) != "EXT":
            return False
        ts = row["sig_entry_ts"] if "sig_entry_ts" in row.index else row["entry_ts"]
        return to_ny(ts).strftime("%H:%M") == "10:25"
    raise ValueError(apply_to)


def _verdict(
    row: dict[str, Any],
    base: dict[str, Any],
    *,
    min_retain_pass: float,
    min_retain_weak: float,
    maxdd_slack: float,
    playbook_improve: float,
) -> str:
    """PASS / WEAK / FAIL vs baseline on primary dual + playbooks."""
    dual_ok = True
    for w in PRIMARY_DUAL:
        n = int(row.get(f"{w}_n") or 0)
        if n == 0:
            continue
        mean = row.get(f"{w}_mean")
        add = float(row.get(f"{w}_add") or 0.0)
        if mean is None or mean <= 0 or add <= 0:
            dual_ok = False
            break

    base_add = float(base.get("dual_add") or 0.0)
    var_add = float(row.get("dual_add") or 0.0)
    retain = (var_add / base_add) if abs(base_add) > 1e-12 else (1.0 if var_add >= 0 else 0.0)

    base_dd = float(base.get("dual_maxdd") or 0.0)
    var_dd = float(row.get("dual_maxdd") or 0.0)
    # maxdd is ≤0; improvement = var_dd closer to 0 (larger algebraically)
    dd_ok_pass = var_dd >= base_dd - 1e-9  # not worse
    dd_ok_weak = var_dd >= base_dd - abs(maxdd_slack)

    gb_base = float(base.get("pb_gave_back_loss") or 0.0)
    gb_var = float(row.get("pb_gave_back_loss") or 0.0)
    # loss_pnl is negative; improve means less negative (larger)
    if gb_base < -1e-9:
        gb_improve = (gb_var - gb_base) / abs(gb_base)
    else:
        gb_improve = 0.0
    gb_ok = gb_improve >= playbook_improve - 1e-9 or gb_var >= gb_base - 1e-12

    ng_base = float(base.get("pb_never_green_loss") or 0.0)
    ng_var = float(row.get("pb_never_green_loss") or 0.0)
    if ng_base < -1e-9:
        ng_improve = (ng_var - ng_base) / abs(ng_base)
    else:
        ng_improve = 0.0
    # only require never_green improve if variant uses confirm
    uses_ca = bool(row.get("uses_confirm"))
    ng_ok = (not uses_ca) or ng_improve >= playbook_improve - 1e-9 or ng_var >= ng_base - 1e-12

    row["retain_dual"] = float(retain)
    row["dd_delta"] = float(var_dd - base_dd)
    row["pb_gave_back_improve"] = float(gb_improve)
    row["pb_never_green_improve"] = float(ng_improve)

    if not dual_ok:
        return "FAIL"
    if retain >= min_retain_pass and dd_ok_pass and gb_ok and ng_ok:
        return "PASS"
    if retain >= min_retain_weak and dd_ok_weak and (gb_ok or retain >= min_retain_pass):
        return "WEAK"
    return "FAIL"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_exit_stress_gate_20260728")
    ap.add_argument("--tp", type=float, default=0.15)
    ap.add_argument("--sl", type=float, default=0.20)
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--max-lag-sec", type=float, default=5.0)
    ap.add_argument("--max-spread-pct", type=float, default=0.15)
    ap.add_argument("--position-frac", type=float, default=0.20)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--variants", default=DEFAULT_VARIANTS)
    ap.add_argument("--min-retain-pass", type=float, default=0.95)
    ap.add_argument("--min-retain-weak", type=float, default=0.90)
    ap.add_argument("--maxdd-slack", type=float, default=0.02)
    ap.add_argument("--playbook-improve", type=float, default=0.05)
    ap.add_argument("--entry-frac", type=float, default=0.75)
    ap.add_argument("--exit-frac", type=float, default=0.75)
    ap.add_argument(
        "--stress",
        action="store_true",
        help="Also replay with lag=8 / spread=0.20 and attach stress_retain",
    )
    ap.add_argument(
        "--sources",
        default="",
        help="Optional override: lane:path,lane:path,... (default=jan_mar+apr+may_jul FO)",
    )
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    quote_root = Path(prof["_paths"]["quote_1s_root"])
    out = Path(prof["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    if args.sources.strip():
        sources: list[tuple[str, str]] = []
        for chunk in args.sources.split(","):
            lane, path = chunk.split(":", 1)
            sources.append((lane.strip().upper(), path.strip()))
    else:
        sources = list(DEFAULT_SOURCES)

    entries = _load_entries(sources)
    fill = FillSpec(entry_frac=args.entry_frac, exit_frac=args.exit_frac)

    variant_specs = [x.strip() for x in args.variants.split(",") if x.strip()]
    variants: list[tuple[str, ExitStressPolicy, str]] = [
        _parse_variant(v, tp=args.tp, sl=args.sl) for v in variant_specs
    ]

    path_cache: dict[tuple[str, str], pd.DataFrame | None] = {}
    day_cache: dict[str, dict[str, pd.DataFrame | None]] = {}

    def get_path(date: str, symbol: str, ticker: str) -> pd.DataFrame | None:
        key = (date, ticker)
        if key in path_cache:
            return path_cache[key]
        if date not in day_cache:
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

    def sim_one(
        er: pd.Series,
        pol: ExitStressPolicy,
        *,
        apply_to: str,
        lag: float,
        spread: float,
    ) -> dict[str, Any] | None:
        use_pol = pol
        if pol.confirm_enabled and not _apply_gate(er, apply_to):
            # scoped confirm off → fall back to same policy without confirm
            from dataclasses import replace as dc_replace

            use_pol = dc_replace(
                pol,
                confirm_enabled=False,
                name=pol.name + "_noca",
            )
        path = get_path(str(er["date"]), str(er["symbol"]), str(er["ticker"]))
        if path is None or path.empty:
            return None
        arm_within = 180.0 if "fast_lad" in pol.name else None
        return simulate_quote_exit_stress(
            path,
            to_ny(er["entry_ts"]),
            use_pol,
            fill=fill,
            max_lag_sec=lag,
            max_spread_pct=spread,
            arm_within_sec=arm_within,
        )

    # ---- baseline tpsl first (for playbook labels + retain) ----
    print(f"entries={len(entries)} variants={len(variants)} out={out}", flush=True)
    base_pol = policy_preset("tpsl", tp=args.tp, sl=args.sl)
    base_rows: list[dict[str, Any]] = []
    skip = 0
    for _, er in entries.iterrows():
        sim = sim_one(er, base_pol, apply_to="all", lag=args.max_lag_sec, spread=args.max_spread_pct)
        if sim is None:
            skip += 1
            continue
        pb = _classify_playbook(float(sim["mfe"]), float(sim["ret"]), str(er["lane"]), sim["entry_ts"])
        base_rows.append(
            {
                "date": str(er["date"]),
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
                "playbook": pb,
                "variant": "tpsl",
                "stress": "nominal",
            }
        )
    base_raw = pd.DataFrame(base_rows)
    base_book = _stack_fixed(
        base_raw, position_frac=args.position_frac, max_concurrent=args.max_concurrent
    )
    print(f"baseline ok={len(base_raw)} skip={skip} book={len(base_book)}", flush=True)

    # playbook key sets from baseline book losers
    base_book = base_book.copy()
    if not base_book.empty:
        base_book["trade_key"] = base_book.apply(_trade_key, axis=1)
    pb_keys = {
        "gave_back": set(
            base_book.loc[base_book["playbook"] == "gave_back", "trade_key"].astype(str)
        )
        if not base_book.empty
        else set(),
        "never_green": set(
            base_book.loc[base_book["playbook"] == "never_green", "trade_key"].astype(str)
        )
        if not base_book.empty
        else set(),
        "open_toxic": set(
            base_book.loc[
                (base_book["playbook"] == "open_toxic") & (base_book["ret"] <= 0),
                "trade_key",
            ].astype(str)
        )
        if not base_book.empty
        else set(),
    }

    def playbook_loss(df: pd.DataFrame, keys: set[str]) -> float:
        if df is None or df.empty or not keys:
            return 0.0
        tk = df.apply(_trade_key, axis=1)
        sub = df[tk.isin(keys)]
        return float(sub["pnl_frac"].sum()) if len(sub) else 0.0

    def dual_metrics(df: pd.DataFrame) -> dict[str, float]:
        parts = []
        for w in PRIMARY_DUAL:
            a, b = next((x[1], x[2]) for x in WINDOWS if x[0] == w)
            parts.append(_window_slice(df, a, b))
        dual = pd.concat([p for p in parts if p is not None and not p.empty], ignore_index=True) if any(
            p is not None and not p.empty for p in parts
        ) else pd.DataFrame()
        s = _summarize(dual)
        return {
            "dual_n": float(s["n"]),
            "dual_add": float(s["add"]),
            "dual_win": float(s["win"]) if s["win"] is not None else float("nan"),
            "dual_mean": float(s["mean"]) if s["mean"] is not None else float("nan"),
            "dual_maxdd": float(s["maxdd"]),
            "dual_compound": float(s["compound"]),
        }

    base_sum = _summarize(base_book)
    base_dual = dual_metrics(base_book)
    base_gate = {
        **base_sum,
        **base_dual,
        "pb_gave_back_loss": playbook_loss(base_book, pb_keys["gave_back"]),
        "pb_never_green_loss": playbook_loss(base_book, pb_keys["never_green"]),
        "pb_open_toxic_loss": playbook_loss(base_book, pb_keys["open_toxic"]),
        "uses_confirm": False,
        "variant": "tpsl",
    }

    all_trades: list[dict[str, Any]] = [dict(r) for r in base_raw.to_dict(orient="records")]
    score_rows: list[dict[str, Any]] = []

    stress_runs = [("nominal", args.max_lag_sec, args.max_spread_pct)]
    if args.stress:
        # Adverse quote stress: tighter lag/spread (harder fills), not looser.
        stress_runs.append(("tight", 3.0, 0.10))

    for vname, pol, apply_to in variants:
        if vname == "tpsl":
            # already have baseline; score it
            row = {
                "variant": "tpsl",
                "apply_to": "all",
                "uses_confirm": False,
                **base_sum,
                **base_dual,
                "pb_gave_back_loss": base_gate["pb_gave_back_loss"],
                "pb_never_green_loss": base_gate["pb_never_green_loss"],
                "pb_open_toxic_loss": base_gate["pb_open_toxic_loss"],
                "stress_retain": 1.0,
            }
            row["verdict"] = _verdict(
                row,
                base_gate,
                min_retain_pass=args.min_retain_pass,
                min_retain_weak=args.min_retain_weak,
                maxdd_slack=args.maxdd_slack,
                playbook_improve=args.playbook_improve,
            )
            # baseline is reference: force PASS on retain identity
            row["verdict"] = "PASS"
            row["retain_dual"] = 1.0
            row["dd_delta"] = 0.0
            score_rows.append(row)
            continue

        for stress_name, lag, spread in stress_runs:
            rows: list[dict[str, Any]] = []
            for _, er in entries.iterrows():
                # only need to match baseline-ok entries
                sim = sim_one(er, pol, apply_to=apply_to, lag=lag, spread=spread)
                if sim is None:
                    continue
                # align playbook label from baseline if present
                key = f"{er['date']}|{er['symbol']}|{er['lane']}|{er['ticker']}|{er['entry_ts']}"
                base_hit = base_raw[
                    (base_raw["date"].astype(str) == str(er["date"]))
                    & (base_raw["symbol"] == str(er["symbol"]))
                    & (base_raw["lane"] == str(er["lane"]))
                    & (base_raw["ticker"] == str(er["ticker"]))
                    & (base_raw["sig_entry_ts"].astype(str) == str(er["entry_ts"]))
                ]
                pb = (
                    str(base_hit.iloc[0]["playbook"])
                    if len(base_hit)
                    else _classify_playbook(
                        float(sim["mfe"]), float(sim["ret"]), str(er["lane"]), sim["entry_ts"]
                    )
                )
                rows.append(
                    {
                        "date": str(er["date"]),
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
                        "playbook": pb,
                        "variant": vname,
                        "stress": stress_name,
                        "_key": key,
                    }
                )
            raw = pd.DataFrame(rows)
            book = _stack_fixed(
                raw, position_frac=args.position_frac, max_concurrent=args.max_concurrent
            )
            if stress_name == "nominal":
                all_trades.extend(raw.drop(columns=["_key"], errors="ignore").to_dict(orient="records"))
                s = _summarize(book)
                d = dual_metrics(book)
                row = {
                    "variant": vname,
                    "apply_to": apply_to,
                    "uses_confirm": bool(pol.confirm_enabled),
                    **s,
                    **d,
                    "pb_gave_back_loss": playbook_loss(book, pb_keys["gave_back"]),
                    "pb_never_green_loss": playbook_loss(book, pb_keys["never_green"]),
                    "pb_open_toxic_loss": playbook_loss(book, pb_keys["open_toxic"]),
                }
                row["verdict"] = _verdict(
                    row,
                    base_gate,
                    min_retain_pass=args.min_retain_pass,
                    min_retain_weak=args.min_retain_weak,
                    maxdd_slack=args.maxdd_slack,
                    playbook_improve=args.playbook_improve,
                )
                # placeholder; filled after stress loop
                row["stress_retain"] = None
                score_rows.append(row)
                print(
                    f"  {vname}: dual_add={d['dual_add']:+.3f} "
                    f"retain={row.get('retain_dual', float('nan')):.3f} "
                    f"maxdd={d['dual_maxdd']:+.3f} "
                    f"gb_imp={row.get('pb_gave_back_improve', 0):+.2f} "
                    f"→ {row['verdict']}",
                    flush=True,
                )
            else:
                # stress retain vs this variant's nominal dual_add
                nom = next(r for r in score_rows if r["variant"] == vname)
                d = dual_metrics(book)
                nom_add = float(nom.get("dual_add") or 0.0)
                stress_ret = (
                    float(d["dual_add"]) / nom_add if abs(nom_add) > 1e-12 else float("nan")
                )
                nom["stress_retain"] = stress_ret
                nom["stress_dual_add"] = float(d["dual_add"])
                if np.isfinite(stress_ret) and stress_ret < 0.85 and nom["verdict"] == "PASS":
                    nom["verdict"] = "WEAK"
                    nom["verdict_note"] = "stress_retain<0.85"

    score = pd.DataFrame(score_rows)
    trades = pd.DataFrame(all_trades)
    score.to_csv(out / "scoreboard.csv", index=False)
    trades.to_csv(out / "trades_all_variants.csv", index=False)
    base_book.to_csv(out / "trades_baseline_book.csv", index=False)

    summary = {
        "tag": args.tag,
        "n_entries": int(len(base_raw)),
        "n_skip": int(skip),
        "position_frac": args.position_frac,
        "max_concurrent": args.max_concurrent,
        "tp": args.tp,
        "sl": args.sl,
        "windows": [list(w) for w in WINDOWS],
        "primary_dual": list(PRIMARY_DUAL),
        "playbook_counts": {k: len(v) for k, v in pb_keys.items()},
        "baseline": base_gate,
        "gates": {
            "min_retain_pass": args.min_retain_pass,
            "min_retain_weak": args.min_retain_weak,
            "maxdd_slack": args.maxdd_slack,
            "playbook_improve": args.playbook_improve,
        },
        "scoreboard": score_rows,
        "promote": [
            r["variant"]
            for r in score_rows
            if r.get("verdict") == "PASS" and r.get("variant") != "tpsl"
        ],
        "weak_candidates": [r["variant"] for r in score_rows if r.get("verdict") == "WEAK"],
        "note": (
            "Exit stress gate on frozen FO entries. Promote only PASS "
            "(WEAK may be sleeve-scoped, e.g. EXT@10:25 only). "
            "Do not use live days to discover TP/SL."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # compact console table
    cols = [
        "variant",
        "verdict",
        "dual_n",
        "dual_add",
        "retain_dual",
        "dual_maxdd",
        "dd_delta",
        "pb_gave_back_improve",
        "pb_never_green_improve",
        "stress_retain",
    ]
    show = score[[c for c in cols if c in score.columns]].copy()
    print("\n=== exit stress gate ===", flush=True)
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"), flush=True)
    print(f"\npromote={summary['promote']} weak={summary['weak_candidates']}", flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
