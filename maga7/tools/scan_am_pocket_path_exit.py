#!/usr/bin/env python3
"""Path-state exits for AM pocket: stock-adverse / giveback / fail-fast.

On champion multi-gate entries, walk option marks jointly with causal stock 1s
and test whether underlying-state exits lift capture vs fixed TP8/SL15.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_pocket_path_exit \\
    --tag research_am_pocket_path_exit
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
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import to_ny
from maga7.common.session_1s_features import prepare_day_arrays
from maga7.common.stock_path_whipsaw import signed_stock_ret
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_pocket_exit_design import _path_window
from maga7.tools.scan_am_pocket_multi_gate import build_gates
from maga7.tools.scan_am_pocket_risk_optimize import _equity_stats, _month_compounds
from maga7.tools.scan_am_pocket_scaleout import simulate_scaleout
from maga7.tools.scan_session_horizon_foresight import _paths_by_ticker

DEFAULT_ENRICHED = Path(
    "/mnt/s990/data/maga7/results/research_am_pocket_multi_gate/enriched_probes.csv"
)
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")

ENTRY_GATES = ("vd_soft", "vd+cont60+mf100+volr12", "vd+volr12")


def _stock_series(
    arr: dict[str, np.ndarray],
    entry_ts: pd.Timestamp,
    max_hold_sec: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (hold_sec, close) from entry along stock 1s, causal."""
    ts_ns = arr["ts_ns"]
    close = arr["close"]
    t0 = int(to_ny(entry_ts).value)
    i0 = int(np.searchsorted(ts_ns, t0, side="left"))
    if i0 >= len(ts_ns):
        return None
    if (int(ts_ns[i0]) - t0) / 1e9 > 5:
        return None
    end_ns = int(ts_ns[i0]) + int(max_hold_sec) * 1_000_000_000
    i_end = int(np.searchsorted(ts_ns, end_ns, side="right") - 1)
    if i_end <= i0:
        return None
    holds = (ts_ns[i0 : i_end + 1] - ts_ns[i0]) / 1e9
    px = close[i0 : i_end + 1].astype(np.float64)
    return holds.astype(np.float64), px


def _stock_signed_at(
    stock_holds: np.ndarray,
    stock_px: np.ndarray,
    entry_px: float,
    direction: str,
    hold_sec: float,
) -> float:
    j = int(np.searchsorted(stock_holds, hold_sec, side="right") - 1)
    if j < 0:
        j = 0
    return float(signed_stock_ret(float(stock_px[j]), entry_px, direction))


def simulate_path_exit(
    opt_rets: np.ndarray,
    opt_holds: np.ndarray,
    *,
    stock_holds: np.ndarray,
    stock_px: np.ndarray,
    direction: str,
    mode: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Causal option exit with optional stock-state overlays."""
    tp = float(params.get("tp", 9.0))
    sl = float(params.get("sl", 9.0))
    max_h = float(params.get("max_hold", opt_holds[-1] if len(opt_holds) else 900))
    min_hold = float(params.get("min_hold", 0.0))
    stock_adv = float(params.get("stock_adv", 9.0))  # adverse thresh (positive number)
    stock_adv_opt_max = float(params.get("stock_adv_opt_max", 9.0))  # only if opt ret < this
    fail_t = float(params.get("fail_t", 9e9))
    fail_stock = float(params.get("fail_stock", 9.0))
    fail_opt = float(params.get("fail_opt", 9.0))
    stk_arm = float(params.get("stk_arm", 9.0))
    stk_giveback = float(params.get("stk_giveback", 9.0))
    opt_arm = float(params.get("opt_arm", 0.0))
    opt_trail = float(params.get("opt_trail", 9.0))

    entry_px = float(stock_px[0])
    if not np.isfinite(entry_px) or entry_px <= 0:
        return {"ret": float(opt_rets[-1]), "reason": "bad_stock", "hold_sec": float(opt_holds[-1])}

    peak_opt = -1.0
    peak_stk = -1.0
    stk_armed = False
    opt_armed = False

    for i in range(1, len(opt_rets)):
        r = float(opt_rets[i])
        h = float(opt_holds[i])
        if h > max_h:
            return {"ret": float(opt_rets[i - 1]), "reason": "max_hold", "hold_sec": float(opt_holds[i - 1])}

        s = _stock_signed_at(stock_holds, stock_px, entry_px, direction, h)
        peak_opt = max(peak_opt, r)
        peak_stk = max(peak_stk, s)
        if s >= stk_arm:
            stk_armed = True
        if r >= opt_arm:
            opt_armed = True

        # hard option TP/SL always on (except oracle mode)
        if mode != "oracle_opt":
            if r >= tp:
                return {"ret": r, "reason": "tp", "hold_sec": h}
            if r <= -sl:
                return {"ret": r, "reason": "sl", "hold_sec": h}

        if mode == "tpsl_only":
            continue

        if h < min_hold:
            continue

        # stock adverse cut (signed stock ret <= -stock_adv)
        if mode in {"stock_adv", "dual", "fail_fast", "stk_gb", "hybrid"}:
            if s <= -stock_adv and r < stock_adv_opt_max:
                return {"ret": r, "reason": "stock_adv", "hold_sec": h}

        # fail-fast: still red stock+opt after T
        if mode in {"fail_fast", "hybrid"} and h >= fail_t:
            if s <= -fail_stock and r < fail_opt:
                return {"ret": r, "reason": "fail_fast", "hold_sec": h}

        # stock giveback after arm
        if mode in {"stk_gb", "hybrid"} and stk_armed and (peak_stk - s) >= stk_giveback:
            return {"ret": r, "reason": "stk_giveback", "hold_sec": h}

        # option trail after arm (optional)
        if mode in {"opt_trail", "hybrid"} and opt_armed and (peak_opt - r) >= opt_trail:
            return {"ret": r, "reason": "opt_trail", "hold_sec": h}

    return {"ret": float(opt_rets[-1]), "reason": "max_hold", "hold_sec": float(opt_holds[-1])}


def _exit_grid() -> list[dict[str, Any]]:
    cfgs: list[dict[str, Any]] = []
    # baselines
    for tp, sl, h in ((0.08, 0.15, 240), (0.10, 0.15, 300), (0.12, 0.15, 300)):
        cfgs.append({"name": f"tpsl_{tp:g}_{sl:g}_h{h}", "mode": "tpsl", "tp": tp, "sl": sl, "max_hold": h})

    # stock adverse overlays on TP8/SL15
    for adv in (0.001, 0.0015, 0.002, 0.003, 0.005):  # 10–50bp adverse
        for minh in (15, 30, 60):
            for opt_max in (0.05, 0.08, 9.0):  # only cut if opt not already up much
                cfgs.append(
                    {
                        "name": f"sadv{adv:g}_mh{minh}_om{opt_max:g}_tp8sl15",
                        "mode": "stock_adv",
                        "tp": 0.08,
                        "sl": 0.15,
                        "max_hold": 300,
                        "min_hold": minh,
                        "stock_adv": adv,
                        "stock_adv_opt_max": opt_max,
                    }
                )

    # fail-fast
    for ft in (30, 60, 90, 120):
        for fs in (0.0, 0.0005, 0.001):
            for fo in (0.0, 0.02):
                cfgs.append(
                    {
                        "name": f"fail_t{ft}_fs{fs:g}_fo{fo:g}_tp8sl15",
                        "mode": "fail_fast",
                        "tp": 0.08,
                        "sl": 0.15,
                        "max_hold": 300,
                        "min_hold": 15,
                        "stock_adv": 0.003,  # also allow mid-hold adverse
                        "stock_adv_opt_max": 0.05,
                        "fail_t": ft,
                        "fail_stock": fs,
                        "fail_opt": fo,
                    }
                )

    # stock giveback
    for arm in (0.0015, 0.002, 0.003):
        for gb in (0.001, 0.0015, 0.002):
            cfgs.append(
                {
                    "name": f"stkgb_a{arm:g}_g{gb:g}_tp8sl15",
                    "mode": "stk_gb",
                    "tp": 0.08,
                    "sl": 0.15,
                    "max_hold": 300,
                    "min_hold": 20,
                    "stk_arm": arm,
                    "stk_giveback": gb,
                    "stock_adv": 0.004,
                    "stock_adv_opt_max": 0.05,
                }
            )

    # hybrid: mild stock adv + opt trail after arm + soft tp
    for adv, trail, arm in (
        (0.002, 0.06, 0.08),
        (0.002, 0.08, 0.10),
        (0.003, 0.06, 0.08),
        (0.0015, 0.05, 0.06),
    ):
        cfgs.append(
            {
                "name": f"hyb_adv{adv:g}_tr{trail:g}_a{arm:g}",
                "mode": "hybrid",
                "tp": 0.25,
                "sl": 0.15,
                "max_hold": 600,
                "min_hold": 20,
                "stock_adv": adv,
                "stock_adv_opt_max": 0.08,
                "opt_arm": arm,
                "opt_trail": trail,
                "stk_arm": 0.002,
                "stk_giveback": 0.002,
                "fail_t": 90,
                "fail_stock": 0.0,
                "fail_opt": 0.0,
            }
        )

    # scale-out reference on same entry
    cfgs.append(
        {
            "name": "scale_ref_67@6_trail",
            "mode": "scale_ref",
            "frac1": 0.67,
            "tp1": 0.06,
            "sl": 0.15,
            "max_hold": 600,
            "runner": "trail",
            "arm": 0.15,
            "trail": 0.15,
            "be_after_scale": True,
            "floor": 0.0,
        }
    )
    return cfgs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--enriched", default=str(DEFAULT_ENRICHED))
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--tag", default="research_am_pocket_path_exit")
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
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    out = Path(prof["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    stock_1s = Path(prof["_paths"]["stock_1s_root"])
    trades_root = Path(args.trades_root)

    probes = pd.read_csv(args.enriched)
    probes["entry_ts"] = pd.to_datetime(probes["entry_ts"])
    probes = probes[probes["enrich_ok"] == True].copy()  # noqa: E712

    gate_map = dict(build_gates())
    path_cache: dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    stock_cache: dict[tuple[str, str], dict[str, np.ndarray] | None] = {}

    def opt_paths(date: str, sym: str):
        key = (date, sym)
        if key not in path_cache:
            tday = load_option_trades(trades_root, sym, date)
            path_cache[key] = (
                _paths_by_ticker(tday) if tday is not None and not tday.empty else {}
            )
        return path_cache[key]

    def stock_arr(date: str, sym: str):
        key = (date, sym)
        if key not in stock_cache:
            raw = load_stock_1s_day(stock_1s, sym, date)
            stock_cache[key] = prepare_day_arrays(raw) if raw is not None and not raw.empty else None
        return stock_cache[key]

    prepared: list[dict[str, Any]] = []
    for _, r in probes.iterrows():
        date, sym = str(r["date"]), str(r["symbol"])
        arrs = opt_paths(date, sym)
        arr = arrs.get(str(r["ticker"]).replace("O:", ""))
        if arr is None:
            continue
        et = to_ny(pd.Timestamp(r["entry_ts"]))
        win = _path_window(arr[0], arr[1], et, max_hold_sec=900, slip=float(args.slip))
        if win is None:
            continue
        rets, holds, _, _ = win
        sarr = stock_arr(date, sym)
        if sarr is None:
            continue
        sw = _stock_series(sarr, et, 900)
        if sw is None:
            continue
        prepared.append(
            {
                "row": r,
                "date": date,
                "symbol": sym,
                "dir": str(r["dir"]),
                "session": str(r["session"]),
                "calendar": str(r["calendar"]),
                "entry_ts": et,
                "rets": rets,
                "holds": holds,
                "stock_holds": sw[0],
                "stock_px": sw[1],
                "oracle_ret": float(r["oracle_ret"]),
            }
        )
    print(f"prepared={len(prepared)}", flush=True)

    entry_masks = {
        g: np.array([bool(gate_map[g](p["row"])) for p in prepared], dtype=bool)
        for g in ENTRY_GATES
    }
    for g, m in entry_masks.items():
        print(f"  entry {g}: n={int(m.sum())}", flush=True)

    exits = _exit_grid()
    print(f"exits={len(exits)}", flush=True)

    score_rows: list[dict[str, Any]] = []
    for gname, mask in entry_masks.items():
        subset = [p for p, ok in zip(prepared, mask) if ok]
        for ex in exits:
            raw = []
            for p in subset:
                if ex["mode"] == "scale_ref":
                    sim = simulate_scaleout(
                        p["rets"],
                        p["holds"],
                        frac1=float(ex["frac1"]),
                        tp1=float(ex["tp1"]),
                        sl=float(ex["sl"]),
                        max_hold=float(ex["max_hold"]),
                        runner=str(ex["runner"]),
                        arm=float(ex["arm"]),
                        trail=float(ex["trail"]),
                        floor=float(ex["floor"]),
                        be_after_scale=True,
                    )
                elif ex["mode"] == "tpsl":
                    sim = simulate_path_exit(
                        p["rets"],
                        p["holds"],
                        stock_holds=p["stock_holds"],
                        stock_px=p["stock_px"],
                        direction=p["dir"],
                        mode="tpsl_only",
                        params=ex,
                    )
                else:
                    sim = simulate_path_exit(
                        p["rets"],
                        p["holds"],
                        stock_holds=p["stock_holds"],
                        stock_px=p["stock_px"],
                        direction=p["dir"],
                        mode=str(ex["mode"]),
                        params=ex,
                    )
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
                # reason mix
                reasons = pd.Series([t["exit_reason"] for t in raw]).value_counts(normalize=True)
                reason_stock = float(reasons.get("stock_adv", 0) + reasons.get("fail_fast", 0) + reasons.get("stk_giveback", 0))
            else:
                mean_cap = float("nan")
                reason_stock = 0.0
            row: dict[str, Any] = {
                "entry": gname,
                "exit": ex["name"],
                "mode": ex["mode"],
                "n_raw": len(raw),
                "mean_capture": mean_cap,
                "frac_stock_exit": reason_stock,
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

    base = sb[(sb.entry == "vd+cont60+mf100+volr12") & (sb.exit == "tpsl_0.08_0.15_h240")]
    base_row = base.iloc[0].to_dict() if len(base) else {}
    bw = float(base_row.get("disc_trade_win") or 0.74)
    bdd = float(base_row.get("disc_maxdd") or -0.12)
    bcap = float(base_row.get("mean_capture") or 0.13)
    bcmp = float(base_row.get("disc_compound") or 0.44)

    soft = sb[
        (sb.entry == "vd+cont60+mf100+volr12")
        & (sb["disc_n"] >= 18)
        & (sb["disc_trade_win"] >= 0.68)
        & (sb["disc_maxdd"] >= -0.16)
        & (sb["disc_compound"] > 0)
        & (sb["may"] > 0)
    ].copy()
    if not soft.empty:
        soft["score"] = (
            soft["disc_trade_win"] * 0.2
            + (1 + soft["disc_maxdd"]) * 0.25
            + np.clip(soft["disc_compound"], 0, 1.5) / 1.5 * 0.2
            + np.clip(soft["mean_capture"], 0, 0.3) / 0.3 * 0.35
        )
        soft = soft.sort_values("score", ascending=False)

    big_cap = sb[
        (sb.entry == "vd+cont60+mf100+volr12")
        & (sb["mean_capture"] >= bcap * 1.25)
        & (sb["disc_trade_win"] >= bw - 0.05)
        & (sb["disc_maxdd"] >= -0.20)
        & (sb["disc_compound"] > 0)
    ].sort_values("mean_capture", ascending=False)

    better = sb[
        (sb.entry == "vd+cont60+mf100+volr12")
        & (sb["disc_trade_win"] >= bw - 0.02)
        & (
            (sb["mean_capture"] > bcap + 0.02)
            | (sb["disc_maxdd"] > bdd + 0.02)
            | (sb["disc_compound"] > bcmp + 0.05)
        )
        & (sb["disc_compound"] > 0)
        & (sb["may"] > 0)
    ].sort_values(["mean_capture", "disc_maxdd"], ascending=[False, False])

    verdict = {
        "protocol": "stock_path_state_exits_on_multi_gate",
        "baseline": base_row,
        "top_soft": soft.head(12).to_dict(orient="records") if len(soft) else [],
        "big_capture": big_cap.head(10).to_dict(orient="records") if len(big_cap) else [],
        "better_than_champ_tpsl": better.head(12).to_dict(orient="records") if len(better) else [],
        "n_exits": len(exits),
    }
    (out / "summary.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")

    cols = [
        c
        for c in [
            "entry",
            "exit",
            "mode",
            "disc_n",
            "disc_trade_win",
            "disc_maxdd",
            "disc_compound",
            "mean_capture",
            "frac_stock_exit",
            "may",
            "jun",
            "jul",
            "blind_trade_win",
            "blind_compound",
        ]
        if c in sb.columns
    ]
    print("\nBASELINE champ+TP8", flush=True)
    print(base[cols].to_string(index=False), flush=True)
    print("\nTOP soft (champ entry)", flush=True)
    print(soft[cols].head(15).to_string(index=False) if len(soft) else "(none)", flush=True)
    print("\nBIG capture (+25% vs base)", flush=True)
    print(big_cap[cols].head(10).to_string(index=False) if len(big_cap) else "(none)", flush=True)
    print("\nBETTER than champ TP8", flush=True)
    print(better[cols].head(12).to_string(index=False) if len(better) else "(none)", flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
