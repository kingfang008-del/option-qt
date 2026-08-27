#!/usr/bin/env python3
"""Multi-indicator entry gates on AM foresight pockets.

Hypothesis: foresight edge is high but vd_soft is too thin — stack causal
1s features (MF/streak/vol/ret/QQQ agree) to raise win / cut DD / lift capture.

Protocol:
  1) Align probes + no_b_up pockets
  2) Enrich with session_1s features_at + QQQ from_open align
  3) Sweep AND / score>=k gates vs vd_soft baseline
  4) Fixed exit TP8/SL15/h240 @20%/5; also report oracle capture

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_pocket_multi_gate \\
    --tag research_am_pocket_multi_gate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.config import load_profile
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import to_ny
from maga7.common.session_1s_features import features_at, prepare_day_arrays
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_pocket_exit_design import (
    ENTRY_VD_SOFT,
    _path_window,
    simulate_exit,
)
from maga7.tools.scan_am_pocket_risk_optimize import (
    POCKET_SETS,
    _entry_ok,
    _equity_stats,
    _month_compounds,
    _signed,
)
from maga7.tools.scan_session_horizon_foresight import _paths_by_ticker

DEFAULT_PROBES = Path(
    "/mnt/s990/data/maga7/results/research_am_vwap_foresight_map_may_jul/probes.csv"
)
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")

EXIT_BASE = {"name": "tpsl_tp0.08_sl0.15_h240", "mode": "tpsl", "tp": 0.08, "sl": 0.15, "max_hold": 240}
EXIT_SCALE = {
    "name": "sc0.67@0.06_tr_a0.15_t0.15_sl0.15_h600",
    "mode": "scale",  # handled separately via scaleout sim if needed; use tpsl proxy first
    "tp": 0.08,
    "sl": 0.15,
    "max_hold": 240,
}


def _s(row: pd.Series, feat: str) -> float:
    """Direction-signed feature (positive = favorable for trade dir)."""
    return _signed(row, feat)


def enrich_probes(
    probes: pd.DataFrame,
    *,
    stock_1s_root: Path,
    symbols: list[str],
) -> pd.DataFrame:
    """Attach causal 1s features + QQQ align at each probe entry_ts."""
    days = sorted(probes["date"].astype(str).unique())
    arr_cache: dict[tuple[str, str], dict[str, np.ndarray] | None] = {}

    def arr_for(date: str, sym: str) -> dict[str, np.ndarray] | None:
        key = (date, sym)
        if key not in arr_cache:
            raw = load_stock_1s_day(stock_1s_root, sym, date)
            if raw is None or getattr(raw, "empty", True):
                arr_cache[key] = None
            else:
                arr_cache[key] = prepare_day_arrays(raw)
        return arr_cache[key]

    extra_rows: list[dict[str, Any]] = []
    for idx, r in probes.iterrows():
        date = str(r["date"])
        sym = str(r["symbol"]).upper()
        et = to_ny(pd.Timestamp(r["entry_ts"]))
        arr = arr_for(date, sym)
        feat = features_at(arr, et) if arr is not None else None
        qarr = arr_for(date, "QQQ")
        qfeat = features_at(qarr, et) if qarr is not None else None
        row: dict[str, Any] = {"_i": idx}
        if feat is None:
            row["enrich_ok"] = False
            extra_rows.append(row)
            continue
        row["enrich_ok"] = True
        for k in (
            "vol_z",
            "volume_ratio_60",
            "mf100",
            "mf300",
            "streak_up",
            "streak_dn",
            "ret_30",
            "ret_60",
            "ret_120",
            "range_60",
            "ret_div_60",
            "from_open",
        ):
            row[k] = feat.get(k)
        # signed helpers stored raw; gate uses dir
        row["qqq_from_open"] = (qfeat or {}).get("from_open")
        # agree: short VWAP FO same sign as dir
        fo10 = float(r.get("fo_vwap10") or np.nan)
        fo30 = float(r.get("fo_vwap30") or np.nan)
        dsign = 1.0 if str(r["dir"]) == "UP" else -1.0
        row["agree_10_30"] = (
            float(np.sign(fo10) == np.sign(fo30) == np.sign(dsign))
            if np.isfinite(fo10) and np.isfinite(fo30) and fo10 != 0 and fo30 != 0
            else 0.0
        )
        row["fo10_abs"] = abs(fo10) if np.isfinite(fo10) else np.nan
        row["fo30_abs"] = abs(fo30) if np.isfinite(fo30) else np.nan
        extra_rows.append(row)
        if len(extra_rows) % 500 == 0:
            print(f"  enrich {len(extra_rows)}/{len(probes)}", flush=True)

    extra = pd.DataFrame(extra_rows).set_index("_i")
    out = probes.copy()
    for c in extra.columns:
        out[c] = extra[c]
    return out


def _fav_streak(row: pd.Series) -> float:
    if str(row["dir"]) == "UP":
        return float(row.get("streak_up") or 0)
    return float(row.get("streak_dn") or 0)


def _signed_feat(row: pd.Series, name: str) -> float:
    v = float(row.get(name) or np.nan)
    if not np.isfinite(v):
        return float("nan")
    return v if str(row["dir"]) == "UP" else -v


GateFn = Callable[[pd.Series], bool]


def build_gates() -> list[tuple[str, GateFn]]:
    """Named causal gates. Positive signed feats = favorable."""

    def vd_soft(r: pd.Series) -> bool:
        return _entry_ok(r, spec=ENTRY_VD_SOFT)

    def accel0(r: pd.Series) -> bool:
        a = float(r.get("accel_10_30") or np.nan)
        return np.isfinite(a) and a >= 0.0

    def agree(r: pd.Series) -> bool:
        return float(r.get("agree_10_30") or 0) >= 1.0

    def cont60(r: pd.Series) -> bool:
        return _signed_feat(r, "ret_60") > 0.0

    def cont30(r: pd.Series) -> bool:
        return _signed_feat(r, "ret_30") > 0.0

    def mf100_pos(r: pd.Series) -> bool:
        return _signed_feat(r, "mf100") > 0.0

    def mf300_pos(r: pd.Series) -> bool:
        return _signed_feat(r, "mf300") > 0.0

    def streak3(r: pd.Series) -> bool:
        return _fav_streak(r) >= 3

    def streak8(r: pd.Series) -> bool:
        return _fav_streak(r) >= 8

    def volr12(r: pd.Series) -> bool:
        v = float(r.get("volume_ratio_60") or np.nan)
        return np.isfinite(v) and v >= 1.2

    def volr15(r: pd.Series) -> bool:
        v = float(r.get("volume_ratio_60") or np.nan)
        return np.isfinite(v) and v >= 1.5

    def volz15(r: pd.Series) -> bool:
        v = float(r.get("vol_z") or np.nan)
        return np.isfinite(v) and v >= 1.5

    def qqq_align(r: pd.Series) -> bool:
        return _signed_feat(r, "qqq_from_open") > 0.0

    def qqq_align_soft(r: pd.Series) -> bool:
        # allow flat QQQ
        v = _signed_feat(r, "qqq_from_open")
        return np.isfinite(v) and v >= -0.001

    def ret_div_pos(r: pd.Series) -> bool:
        return _signed_feat(r, "ret_div_60") > 0.0

    def not_blow(r: pd.Series) -> bool:
        fo = _signed(r, "fo_vwap30")
        return np.isfinite(fo) and fo <= 0.012

    def mild_fo(r: pd.Series) -> bool:
        fo = _signed(r, "fo_vwap30")
        return np.isfinite(fo) and 0.003 <= fo <= 0.015

    def mild_vd(r: pd.Series) -> bool:
        vd = _signed(r, "vwap_diff")
        return np.isfinite(vd) and 0.002 <= vd <= 0.007

    atoms: list[tuple[str, GateFn]] = [
        ("vd_soft", vd_soft),
        ("accel0", accel0),
        ("agree", agree),
        ("cont30", cont30),
        ("cont60", cont60),
        ("mf100+", mf100_pos),
        ("mf300+", mf300_pos),
        ("streak3", streak3),
        ("streak8", streak8),
        ("volr12", volr12),
        ("volr15", volr15),
        ("volz15", volz15),
        ("qqq+", qqq_align),
        ("qqq~", qqq_align_soft),
        ("retdiv+", ret_div_pos),
        ("not_blow", not_blow),
        ("mild_fo", mild_fo),
        ("mild_vd", mild_vd),
    ]

    gates: list[tuple[str, GateFn]] = list(atoms)

    # stacked ANDs on vd_soft
    stacks = [
        ("vd+agree", ["vd_soft", "agree"]),
        ("vd+cont60", ["vd_soft", "cont60"]),
        ("vd+mf100", ["vd_soft", "mf100+"]),
        ("vd+streak3", ["vd_soft", "streak3"]),
        ("vd+volr12", ["vd_soft", "volr12"]),
        ("vd+qqq", ["vd_soft", "qqq+"]),
        ("vd+retdiv", ["vd_soft", "retdiv+"]),
        ("vd+agree+cont60", ["vd_soft", "agree", "cont60"]),
        ("vd+agree+mf100", ["vd_soft", "agree", "mf100+"]),
        ("vd+cont60+mf100", ["vd_soft", "cont60", "mf100+"]),
        ("vd+cont60+qqq", ["vd_soft", "cont60", "qqq+"]),
        ("vd+mf100+volr12", ["vd_soft", "mf100+", "volr12"]),
        ("vd+agree+cont60+mf100", ["vd_soft", "agree", "cont60", "mf100+"]),
        ("vd+agree+cont60+qqq", ["vd_soft", "agree", "cont60", "qqq+"]),
        ("vd+cont60+mf100+volr12", ["vd_soft", "cont60", "mf100+", "volr12"]),
        (
            "vd+agree+cont60+mf100+qqq",
            ["vd_soft", "agree", "cont60", "mf100+", "qqq+"],
        ),
        (
            "vd+agree+cont60+mf100+volr12",
            ["vd_soft", "agree", "cont60", "mf100+", "volr12"],
        ),
        (
            "full5",
            ["vd_soft", "agree", "cont60", "mf100+", "qqq+", "volr12"],
        ),
        # without vd_soft — structure-only
        ("struct3", ["agree", "cont60", "mf100+"]),
        ("struct4", ["agree", "cont60", "mf100+", "qqq+"]),
        ("struct4v", ["agree", "cont60", "mf100+", "volr12"]),
        ("struct5", ["agree", "cont60", "mf100+", "qqq+", "volr12"]),
        ("mild+struct", ["mild_fo", "mild_vd", "agree", "cont60", "mf100+"]),
    ]
    atom_map = dict(atoms)

    def _and(names: list[str]) -> GateFn:
        fns = [atom_map[n] for n in names]

        def g(r: pd.Series, _fns=fns) -> bool:
            return all(f(r) for f in _fns)

        return g

    for name, parts in stacks:
        gates.append((name, _and(parts)))

    # score >= k over a feature pack (excluding vd_soft as required base optional)
    pack = ["agree", "cont30", "cont60", "mf100+", "mf300+", "streak3", "volr12", "qqq+", "retdiv+", "not_blow"]
    pack_fns = [atom_map[n] for n in pack]

    def _score_ge(k: int, require_vd: bool) -> GateFn:
        def g(r: pd.Series, _k=k, _req=require_vd) -> bool:
            if _req and not vd_soft(r):
                return False
            s = sum(1 for f in pack_fns if f(r))
            return s >= _k

        return g

    for k in (3, 4, 5, 6):
        gates.append((f"score{k}", _score_ge(k, require_vd=False)))
        gates.append((f"vd+score{k}", _score_ge(k, require_vd=True)))

    return gates


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", default=str(DEFAULT_PROBES))
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--tag", default="research_am_pocket_multi_gate")
    ap.add_argument(
        "--profile",
        default=(
            "maga7/CONFIG/strategy_profiles/"
            "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
        ),
    )
    ap.add_argument("--position-frac", type=float, default=0.20)
    ap.add_argument("--max-concurrent", type=int, default=5)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--skip-enrich", action="store_true", help="reuse enriched csv if present")
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    out = Path(prof["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    stock_1s = Path(prof["_paths"]["stock_1s_root"])
    trades_root = Path(args.trades_root)
    symbols = list(prof.get("symbols") or [])

    enriched_path = out / "enriched_probes.csv"
    if args.skip_enrich and enriched_path.exists():
        print(f"load enriched {enriched_path}", flush=True)
        probes = pd.read_csv(enriched_path)
        probes["entry_ts"] = pd.to_datetime(probes["entry_ts"])
    else:
        probes = pd.read_csv(args.probes)
        probes["entry_ts"] = pd.to_datetime(probes["entry_ts"])
        probes = probes[
            probes["dir"] == np.where(probes["from_open_px"].astype(float) >= 0, "UP", "DN")
        ].copy()
        pdf = pd.DataFrame(
            sorted(POCKET_SETS["no_b_up"]), columns=["session", "tod_bucket", "dir"]
        )
        probes = probes.merge(pdf, on=["session", "tod_bucket", "dir"], how="inner")
        probes = probes.sort_values(["date", "symbol", "session", "entry_ts"]).drop_duplicates(
            ["date", "symbol", "session"], keep="first"
        )
        print(f"pocket probes={len(probes)} enriching…", flush=True)
        probes = enrich_probes(probes, stock_1s_root=stock_1s, symbols=symbols)
        probes.to_csv(enriched_path, index=False)
        print(f"wrote {enriched_path} enrich_ok={int(probes['enrich_ok'].sum())}", flush=True)

    probes = probes[probes["enrich_ok"] == True].copy()  # noqa: E712

    # option paths
    path_cache: dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    def paths_for(date: str, sym: str):
        key = (date, sym)
        if key not in path_cache:
            tday = load_option_trades(trades_root, sym, date)
            path_cache[key] = (
                _paths_by_ticker(tday) if tday is not None and not tday.empty else {}
            )
        return path_cache[key]

    prepared: list[dict[str, Any]] = []
    for _, r in probes.iterrows():
        arrs = paths_for(str(r["date"]), str(r["symbol"]))
        arr = arrs.get(str(r["ticker"]).replace("O:", ""))
        if arr is None:
            continue
        win = _path_window(
            arr[0],
            arr[1],
            to_ny(pd.Timestamp(r["entry_ts"])),
            max_hold_sec=900,
            slip=float(args.slip),
        )
        if win is None:
            continue
        rets, holds, _, _ = win
        prepared.append(
            {
                "row": r,
                "date": str(r["date"]),
                "symbol": str(r["symbol"]),
                "dir": str(r["dir"]),
                "session": str(r["session"]),
                "calendar": str(r["calendar"]),
                "entry_ts": to_ny(pd.Timestamp(r["entry_ts"])),
                "rets": rets,
                "holds": holds,
                "oracle_ret": float(r["oracle_ret"]),
            }
        )
    print(f"prepared paths={len(prepared)}", flush=True)

    gates = build_gates()
    # precompute gate mask per prepared
    gate_masks: dict[str, np.ndarray] = {}
    for gname, gfn in gates:
        mask = np.array([bool(gfn(p["row"])) for p in prepared], dtype=bool)
        gate_masks[gname] = mask
        print(f"  gate {gname}: n={int(mask.sum())}", flush=True)

    score_rows: list[dict[str, Any]] = []
    ex = EXIT_BASE
    for gname, mask in gate_masks.items():
        raw = []
        for p, ok in zip(prepared, mask):
            if not ok:
                continue
            mh = float(ex["max_hold"])
            m = p["holds"] <= mh + 1e-9
            rets = p["rets"][m]
            holds = p["holds"][m]
            if len(rets) < 2:
                continue
            sim = simulate_exit(rets, holds, mode=str(ex["mode"]), params=ex)
            if not np.isfinite(sim["ret"]):
                continue
            et = p["entry_ts"]
            raw.append(
                {
                    "date": p["date"],
                    "symbol": p["symbol"],
                    "dir": p["dir"],
                    "session": p["session"],
                    "calendar": p["calendar"],
                    "entry_ts": et,
                    "exit_ts": et + pd.Timedelta(seconds=float(sim["hold_sec"])),
                    "ret": float(sim["ret"]),
                    "exit_reason": str(sim["reason"]),
                    "hold_sec": float(sim["hold_sec"]),
                    "oracle_ret": float(p["oracle_ret"]),
                }
            )
        disc = [t for t in raw if t["calendar"] == "may_jul09"]
        blind = [t for t in raw if t["calendar"] == "jul10_23"]
        sized_d = _portfolio_day(
            sorted(disc, key=lambda x: (x["entry_ts"], x["symbol"])),
            position_frac=float(args.position_frac),
            max_concurrent=int(args.max_concurrent),
            cooldown_minutes=10.0,
        )
        sized_b = _portfolio_day(
            sorted(blind, key=lambda x: (x["entry_ts"], x["symbol"])),
            position_frac=float(args.position_frac),
            max_concurrent=int(args.max_concurrent),
            cooldown_minutes=10.0,
        )
        st_d = _equity_stats(pd.DataFrame(sized_d))
        st_b = _equity_stats(pd.DataFrame(sized_b))
        months = _month_compounds(pd.DataFrame(sized_d + sized_b))
        if raw:
            o = np.array([t["oracle_ret"] for t in raw], dtype=float)
            rr = np.array([t["ret"] for t in raw], dtype=float)
            mean_cap = float(rr.mean() / o.mean()) if o.mean() > 0 else float("nan")
            oracle_win = float((o >= 0.15).mean())
        else:
            mean_cap = oracle_win = float("nan")
        row: dict[str, Any] = {
            "gate": gname,
            "n_raw": len(raw),
            "mean_capture": mean_cap,
            "oracle_edge_rate": oracle_win,
            "may": months.get("2026-05"),
            "jun": months.get("2026-06"),
            "jul": months.get("2026-07"),
        }
        for k, v in st_d.items():
            row[f"disc_{k}"] = v
        for k, v in st_b.items():
            row[f"blind_{k}"] = v
        score_rows.append(row)

    sb = pd.DataFrame(score_rows)
    sb.to_csv(out / "scoreboard.csv", index=False)

    base = sb[sb["gate"] == "vd_soft"]
    base_row = base.iloc[0].to_dict() if len(base) else {}
    bw = float(base_row.get("disc_trade_win") or 0)
    bdd = float(base_row.get("disc_maxdd") or -1)
    bcmp = float(base_row.get("disc_compound") or 0)
    bcap = float(base_row.get("mean_capture") or 0)

    # lifts
    soft = sb[
        (sb["disc_n"] >= 20)
        & (sb["disc_trade_win"] >= 0.62)
        & (sb["disc_maxdd"] >= -0.20)
        & (sb["blind_n"] >= 4)
    ].copy()
    if not soft.empty:
        soft["lift_win"] = soft["disc_trade_win"] - bw
        soft["lift_dd"] = soft["disc_maxdd"] - bdd
        soft["lift_cmp"] = soft["disc_compound"] - bcmp
        soft["lift_cap"] = soft["mean_capture"] - bcap
        soft["score"] = (
            soft["disc_trade_win"] * 0.25
            + (1 + soft["disc_maxdd"]) * 0.25
            + np.clip(soft["disc_compound"], 0, 2) / 2 * 0.2
            + np.clip(soft["mean_capture"], 0, 0.25) / 0.25 * 0.3
        )
        soft = soft.sort_values("score", ascending=False)

    # big improvement: win+5pp OR capture+50% relative OR dd+3pp with win not worse
    big = sb[
        (sb["disc_n"] >= 15)
        & (
            (sb["disc_trade_win"] >= bw + 0.05)
            | (sb["mean_capture"] >= bcap * 1.5)
            | ((sb["disc_maxdd"] >= bdd + 0.03) & (sb["disc_trade_win"] >= bw - 0.02))
        )
        & (sb["disc_compound"] > 0)
    ].sort_values("mean_capture", ascending=False)

    verdict = {
        "protocol": "multi_indicator_gates_on_no_b_up_pockets",
        "exit": EXIT_BASE,
        "portfolio": {"position_frac": args.position_frac, "max_concurrent": args.max_concurrent},
        "baseline_vd_soft": base_row,
        "top_soft": soft.head(10).to_dict(orient="records") if len(soft) else [],
        "big_improve": big.head(10).to_dict(orient="records") if len(big) else [],
        "n_gates": len(gates),
        "n_prepared": len(prepared),
    }
    (out / "summary.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")

    cols = [
        c
        for c in [
            "gate",
            "n_raw",
            "disc_n",
            "disc_trade_win",
            "disc_day_win",
            "disc_maxdd",
            "disc_compound",
            "mean_capture",
            "oracle_edge_rate",
            "may",
            "jun",
            "jul",
            "blind_trade_win",
            "blind_compound",
        ]
        if c in sb.columns
    ]
    print("\nBASELINE vd_soft", flush=True)
    print(sb[sb.gate == "vd_soft"][cols].to_string(index=False), flush=True)
    print("\nTOP by score (soft filter)", flush=True)
    print((soft[cols].head(15) if len(soft) else sb.sort_values("disc_trade_win", ascending=False)[cols].head(15)).to_string(index=False), flush=True)
    print("\nBIG improve candidates", flush=True)
    print(big[cols].head(12).to_string(index=False) if len(big) else "(none)", flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
