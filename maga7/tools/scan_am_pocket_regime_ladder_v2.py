#!/usr/bin/env python3
"""AM pocket v2: trade-only, **widen-only** ladders + stock-confirm upgrade.

vs v1 failures:
  - CHOP no longer tighter than TP8
  - first take never before +8% (no early lock)
  - SL never tighter than 15%; hold never shorter than 240s
  - path adapt = stock-favorable **upgrade only** (no naked option fail-fast)

Mark: option trade-last ±slip. Quotes unused.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_pocket_regime_ladder_v2 \\
    --tag research_am_pocket_regime_ladder_v2
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

from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.config import load_profile
from maga7.common.option_trade_tpsl import simulate_trade_tpsl
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import to_ny
from maga7.common.session_1s_features import prepare_day_arrays
from maga7.common.stock_path_whipsaw import signed_stock_ret
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_delayed_confirm_quote_dual import _ok, _stats
from maga7.tools.scan_am_pocket_exit_design import _path_window
from maga7.tools.scan_am_pocket_multi_gate import build_gates
from maga7.tools.scan_am_pocket_path_exit import _stock_series, _stock_signed_at
from maga7.tools.scan_am_pocket_risk_optimize import POCKET_SETS, _equity_stats, _month_compounds
from maga7.tools.scan_am_pocket_scaleout import simulate_scaleout
from maga7.tools.scan_session_horizon_foresight import _paths_by_ticker

DEFAULT_ENRICHED = Path(
    "/mnt/s990/data/maga7/results/research_am_pocket_multi_gate/enriched_probes.csv"
)
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")
DEFAULT_STOCK = Path("/mnt/s990/data/raw_1s/stocks")
PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
WINDOWS = (
    ("may_jul09", "2026-05-01", "2026-07-09"),
    ("jul10_23", "2026-07-10", "2026-07-23"),
)

# Widen-only vs TP8/SL15/h240 floor.
# v2b: prefer hard tp2 on runner — trail was giving back scale profits.
LADDERS_V2: dict[str, dict[str, Any]] = {
    "CHOP": {"mode": "tpsl", "tp": 0.08, "sl": 0.15, "max_hold": 240},
    "TREND": {
        "mode": "scale",
        "frac1": 0.50,
        "tp1": 0.08,
        "sl": 0.15,
        "max_hold": 480,
        "runner": "tp",
        "arm": 0.0,
        "trail": 9.0,
        "tp2": 0.20,
        "be_after_scale": True,
        "floor": 0.0,
    },
    "IMPULSE": {
        "mode": "scale",
        "frac1": 0.33,
        "tp1": 0.08,
        "sl": 0.15,
        "max_hold": 600,
        "runner": "tp",
        "arm": 0.0,
        "trail": 9.0,
        "tp2": 0.30,
        "be_after_scale": True,
        "floor": 0.0,
    },
}

# Alternate ladder pack (trail after higher arm)
LADDERS_V2_TRAIL: dict[str, dict[str, Any]] = {
    "CHOP": {"mode": "tpsl", "tp": 0.08, "sl": 0.15, "max_hold": 240},
    "TREND": {
        "mode": "scale",
        "frac1": 0.50,
        "tp1": 0.08,
        "sl": 0.15,
        "max_hold": 480,
        "runner": "trail",
        "arm": 0.16,
        "trail": 0.08,
        "tp2": 0.25,
        "be_after_scale": True,
        "floor": 0.0,
    },
    "IMPULSE": {
        "mode": "scale",
        "frac1": 0.33,
        "tp1": 0.08,
        "sl": 0.15,
        "max_hold": 600,
        "runner": "trail",
        "arm": 0.22,
        "trail": 0.10,
        "tp2": 0.35,
        "be_after_scale": True,
        "floor": 0.0,
    },
}


def classify_regime_v2(row: pd.Series) -> str:
    fo = abs(float(row.get("fo_vwap30") or 0.0))
    volr = float(row.get("volume_ratio_60") or 0.0)
    accel = float(row.get("accel_10_30") or 0.0)
    if not np.isfinite(volr):
        volr = 0.0
    if not np.isfinite(accel):
        accel = 0.0
    if volr >= 1.5 and fo >= 0.010 and accel >= 1e-4:
        return "IMPULSE"
    if volr >= 1.2 and fo >= 0.005:
        return "TREND"
    return "CHOP"


def _apply_ladder(
    rets: np.ndarray,
    holds: np.ndarray,
    regime: str,
    *,
    pack: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ladders = pack or LADDERS_V2
    spec = ladders[regime]
    if spec["mode"] == "tpsl":
        tp, sl, mh = float(spec["tp"]), float(spec["sl"]), float(spec["max_hold"])
        for i in range(1, len(rets)):
            r, h = float(rets[i]), float(holds[i])
            if r >= tp:
                return {"ret": r, "reason": "tp", "hold_sec": h, "regime": regime, "scaled": False}
            if r <= -sl:
                return {"ret": r, "reason": "sl", "hold_sec": h, "regime": regime, "scaled": False}
            if h >= mh:
                return {
                    "ret": float(rets[i - 1]),
                    "reason": "max_hold",
                    "hold_sec": float(holds[i - 1]),
                    "regime": regime,
                    "scaled": False,
                }
        return {
            "ret": float(rets[-1]),
            "reason": "max_hold",
            "hold_sec": float(holds[-1]),
            "regime": regime,
            "scaled": False,
        }
    out = simulate_scaleout(
        rets,
        holds,
        frac1=float(spec["frac1"]),
        tp1=float(spec["tp1"]),
        sl=float(spec["sl"]),
        max_hold=float(spec["max_hold"]),
        runner=str(spec["runner"]),
        tp2=float(spec.get("tp2", 9.0)),
        arm=float(spec.get("arm", 0.0)),
        trail=float(spec.get("trail", 9.0)),
        floor=float(spec.get("floor", -9.0)),
        be_after_scale=bool(spec.get("be_after_scale", False)),
    )
    out["regime"] = regime
    return out


def _stock_upgrade(
    *,
    rets: np.ndarray,
    holds: np.ndarray,
    stock_holds: np.ndarray | None,
    stock_px: np.ndarray | None,
    direction: str,
    regime0: str,
    confirm_sec: float = 30.0,
    stock_min: float = 0.0020,
    opt_min: float = 0.01,
) -> tuple[str, str | None]:
    """Upgrade CHOP/TREND → IMPULSE only if stock + option both confirm.

    Defaults from ``research_am_pocket_stock_up_grid`` (vd_soft): confirm=30s,
    stock≥20bp; opt_min is weak (often non-binding once stock confirms).
    """
    if regime0 == "IMPULSE" or stock_holds is None or stock_px is None:
        return regime0, None
    i = int(np.searchsorted(holds, confirm_sec, side="left"))
    i = min(max(i, 1), len(rets) - 1)
    r_opt = float(rets[i])
    entry_px = float(stock_px[0])
    s = _stock_signed_at(stock_holds, stock_px, entry_px, direction, float(holds[i]))
    if r_opt >= opt_min and s >= stock_min:
        return "IMPULSE", "stock_upgrade"
    return regime0, None


def _window_of(date: str) -> str | None:
    for name, a, b in WINDOWS:
        if a <= date <= b:
            return name
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--enriched", default=str(DEFAULT_ENRICHED))
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--stock-1s", default=str(DEFAULT_STOCK))
    ap.add_argument("--tag", default="research_am_pocket_regime_ladder_v2")
    ap.add_argument(
        "--entries",
        default="vd+cont60+mf100+volr12,vd_soft",
        help="comma entry gate names",
    )
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--position-frac", type=float, default=0.20)
    ap.add_argument("--max-concurrent", type=int, default=5)
    ap.add_argument("--cooldown-minutes", type=float, default=10.0)
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    out = Path(prof["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    trades_root = Path(args.trades_root)
    stock_root = Path(args.stock_1s)

    probes_all = pd.read_csv(args.enriched)
    probes_all["entry_ts"] = pd.to_datetime(probes_all["entry_ts"])
    probes_all = probes_all[probes_all["enrich_ok"] == True].copy()  # noqa: E712
    pdf = pd.DataFrame(sorted(POCKET_SETS["no_b_up"]), columns=["session", "tod_bucket", "dir"])
    probes_all = probes_all.merge(pdf, on=["session", "tod_bucket", "dir"], how="inner")
    gate_map = dict(build_gates())

    tcache: dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    scache: dict[tuple[str, str], dict[str, np.ndarray] | None] = {}

    def paths_for(date: str, sym: str):
        key = (date, sym)
        if key not in tcache:
            tday = load_option_trades(trades_root, sym, date)
            tcache[key] = (
                _paths_by_ticker(tday) if tday is not None and not tday.empty else {}
            )
        return tcache[key]

    def stock_arr(date: str, sym: str):
        key = (date, sym)
        if key not in scache:
            raw = load_stock_1s_day(stock_root, sym, date)
            if raw is None or raw.empty:
                scache[key] = None
            else:
                scache[key] = prepare_day_arrays(raw)
        return scache[key]

    variants = (
        # (name, mode, stock_up, pack)
        ("fixed_tp8", "fixed", False, None),
        ("widen_tp2", "regime", False, "tp2"),
        ("widen_tp2_stock_up", "regime", True, "tp2"),
        ("widen_trail", "regime", False, "trail"),
        ("widen_trail_stock_up", "regime", True, "trail"),
        ("all_impulse_tp2", "all_impulse", False, "tp2"),
    )

    score_rows: list[dict[str, Any]] = []
    entries = [x.strip() for x in str(args.entries).split(",") if x.strip()]

    for entry_name in entries:
        gfn = gate_map[entry_name]
        probes = probes_all[probes_all.apply(gfn, axis=1)].copy()
        print(f"\n=== entry={entry_name} n={len(probes)} ===", flush=True)

        for vname, mode, stock_up, pack_name in variants:
            pack = LADDERS_V2 if pack_name in (None, "tp2") else LADDERS_V2_TRAIL
            win_raw: dict[str, list[dict[str, Any]]] = {w[0]: [] for w in WINDOWS}
            regime_counts: dict[str, int] = {"CHOP": 0, "TREND": 0, "IMPULSE": 0, "FIXED": 0}
            n_upgrade = 0
            for _, r in probes.iterrows():
                date, sym = str(r["date"]), str(r["symbol"])
                w = _window_of(date)
                if w is None:
                    continue
                ticker = str(r["ticker"]).replace("O:", "")
                arr = paths_for(date, sym).get(ticker)
                if arr is None:
                    continue
                et = to_ny(pd.Timestamp(r["entry_ts"]))
                direction = str(r["dir"])

                if mode == "fixed":
                    sim = simulate_trade_tpsl(
                        arr[0], arr[1], et, tp=0.08, sl=0.15, max_hold_sec=240, slip=float(args.slip)
                    )
                    if sim is None or not np.isfinite(sim["ret"]):
                        continue
                    reg, hold, ret, reason = "FIXED", float(sim["hold_sec"]), float(sim["ret"]), str(sim["reason"])
                    regime_counts["FIXED"] += 1
                else:
                    pw = _path_window(arr[0], arr[1], et, max_hold_sec=600, slip=float(args.slip))
                    if pw is None:
                        continue
                    rets, holds, _, _ = pw
                    if mode == "all_impulse":
                        reg0 = "IMPULSE"
                    else:
                        reg0 = classify_regime_v2(r)
                    regime_counts[reg0] = regime_counts.get(reg0, 0) + 1
                    tag = None
                    reg = reg0
                    if stock_up:
                        sarr = stock_arr(date, sym)
                        sh = sp = None
                        if sarr is not None:
                            ss = _stock_series(sarr, et, 600)
                            if ss is not None:
                                sh, sp = ss
                        reg, tag = _stock_upgrade(
                            rets=rets,
                            holds=holds,
                            stock_holds=sh,
                            stock_px=sp,
                            direction=direction,
                            regime0=reg0,
                        )
                        if tag:
                            n_upgrade += 1
                    sim = _apply_ladder(rets, holds, reg, pack=pack)
                    if sim is None or not np.isfinite(sim.get("ret", float("nan"))):
                        continue
                    hold, ret = float(sim["hold_sec"]), float(sim["ret"])
                    reason = str(sim["reason"]) + (f"+{tag}" if tag else "")
                    reg = str(sim.get("regime") or reg)

                win_raw[w].append(
                    {
                        "date": date,
                        "symbol": sym,
                        "dir": direction,
                        "session": str(r["session"]),
                        "entry_ts": et,
                        "exit_ts": et + pd.Timedelta(seconds=hold),
                        "ticker": ticker,
                        "ret": ret,
                        "exit_reason": reason,
                        "hold_sec": hold,
                        "oracle_ret": float(r["oracle_ret"]),
                        "regime": reg,
                        "variant": vname,
                        "entry_gate": entry_name,
                        "window": w,
                        "event_source": "am_pocket_regime_ladder_v2",
                    }
                )

            win_stats: dict[str, dict[str, Any]] = {}
            sized_all: list[dict[str, Any]] = []
            for wname, _, _ in WINDOWS:
                raw = win_raw[wname]
                by_d: dict[str, list] = {}
                for tr in raw:
                    by_d.setdefault(str(tr["date"]), []).append(tr)
                sized: list[dict] = []
                for _, rs in sorted(by_d.items()):
                    sized.extend(
                        _portfolio_day(
                            sorted(rs, key=lambda x: (x["entry_ts"], x["symbol"])),
                            position_frac=float(args.position_frac),
                            max_concurrent=int(args.max_concurrent),
                            cooldown_minutes=float(args.cooldown_minutes),
                        )
                    )
                st = _stats(sized)
                st_eq = _equity_stats(pd.DataFrame(sized)) if sized else {}
                for k, v in st_eq.items():
                    st[f"eq_{k}"] = v
                win_stats[wname] = st
                sized_all.extend(sized)

            disc_c = float(win_stats["may_jul09"].get("eq_compound") or 0)
            blind_c = float(win_stats["jul10_23"].get("eq_compound") or 0)
            n_d = int(win_stats["may_jul09"].get("n") or 0)
            n_b = int(win_stats["jul10_23"].get("n") or 0)
            both_econ = bool(n_d >= 8 and n_b >= 3 and disc_c > 0 and blind_c > 0)
            if sized_all:
                o = np.array([t["oracle_ret"] for t in sized_all], dtype=float)
                rr = np.array([t["ret"] for t in sized_all], dtype=float)
                mean_cap = float(rr.mean() / o.mean()) if np.nanmean(o) > 0 else float("nan")
                pd.DataFrame(sized_all).to_csv(
                    out / f"trades_{entry_name.replace('+','_')}_{vname}.csv", index=False
                )
            else:
                mean_cap = float("nan")
            months = _month_compounds(pd.DataFrame(sized_all)) if sized_all else {}
            row = {
                "entry": entry_name,
                "variant": vname,
                "econ_dual_pass": both_econ,
                "n_raw": sum(len(v) for v in win_raw.values()),
                "mean_capture": mean_cap,
                "disc_compound": disc_c,
                "blind_compound": blind_c,
                "disc_n": n_d,
                "blind_n": n_b,
                "n_stock_upgrade": n_upgrade,
                "regime_counts": json.dumps(regime_counts),
                "may": months.get("2026-05"),
                "jun": months.get("2026-06"),
                "jul": months.get("2026-07"),
                "disc_mean": win_stats["may_jul09"].get("mean"),
                "disc_win": win_stats["may_jul09"].get("trade_win"),
                "disc_maxdd": win_stats["may_jul09"].get("eq_maxdd"),
                "blind_mean": win_stats["jul10_23"].get("mean"),
                "blind_win": win_stats["jul10_23"].get("trade_win"),
            }
            score_rows.append(row)
            print(
                f"  {vname}: n={row['n_raw']} disc={disc_c:+.3f} blind={blind_c:+.3f} "
                f"cap={mean_cap:.3f} up={n_upgrade} econ={both_econ}",
                flush=True,
            )

    sb = pd.DataFrame(score_rows)
    sb.to_csv(out / "scoreboard.csv", index=False)

    # promote: same entry, beat fixed_tp8 on capture (or compound with capture≥baseline)
    promotes: list[dict[str, Any]] = []
    for entry_name in entries:
        base = sb[(sb.entry == entry_name) & (sb.variant == "fixed_tp8")]
        if base.empty:
            continue
        b = base.iloc[0]
        for _, r in sb[sb.entry == entry_name].iterrows():
            if r["variant"] == "fixed_tp8" or not r["econ_dual_pass"]:
                continue
            cap_lift = float(r["mean_capture"]) > float(b["mean_capture"]) + 1e-6
            comp_lift = float(r["disc_compound"]) > float(b["disc_compound"]) + 1e-9
            cap_ok = float(r["mean_capture"]) + 1e-9 >= float(b["mean_capture"])
            if cap_lift or (comp_lift and cap_ok):
                promotes.append(r.to_dict())

    promotes.sort(key=lambda x: (float(x["mean_capture"]), float(x["disc_compound"])), reverse=True)
    summary = {
        "protocol": "am_pocket_regime_ladder_v2_widen_only",
        "mark": "option_trade_last_slip",
        "quote": "NOT_USED",
        "ladders_tp2": LADDERS_V2,
        "ladders_trail": LADDERS_V2_TRAIL,
        "promote": "NONE" if not promotes else f"RESEARCH_{promotes[0]['entry']}__{promotes[0]['variant']}",
        "promotes": promotes[:5],
        "scoreboard": score_rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({"promote": summary["promote"], "promotes": summary["promotes"]}, indent=2, default=str))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
